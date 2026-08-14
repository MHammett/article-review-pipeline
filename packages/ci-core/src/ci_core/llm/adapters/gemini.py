"""Gemini fact-check adapter.

Supports two providers:
  - ai_studio  (default) — Google AI Studio REST API, API-key auth.
  - vertex_ai             — Google Cloud Vertex AI, service-account or ADC auth.

Provider is selected via the ``provider_config`` dict passed by the pipeline.
Existing user.yaml files that specify a plain model string continue to work
unchanged; the config_loader normalises them to ``{"provider": "ai_studio",
"model": "..."}`` before they reach this adapter.

Vertex AI requirements
----------------------
Install the optional dependency before switching providers::

    pip install google-auth

Then in configs/user.yaml::

    models:
      gemini:
        provider: vertex_ai
        model: gemini-2.5-flash
        project: my-gcp-project          # GCP project ID
        location: us-central1            # Vertex AI region
        # credentials_file: /path/to/sa.json  # omit to use Application Default Credentials
"""

import time
import logging
import requests

from .. import schema_format, streaming
from ..tokens import normalize_tokens
from ... import redact
from ..json_utils import extract_json_with_salvage as _extract_json_with_salvage

DEFAULT_MODEL = "gemini-2.5-flash"
# Inter-token read-gap timeout (seconds); constant, not the sliding-scale value.
# Grounded fact-check runs a live Google Search before the first token, so allow a
# wider first-byte gap than the non-grounded chat models.
_READ_TIMEOUT = 160
# Fallback chain tried in order when a model returns 503 (capacity unavailable).
# gemini-2.5-flash-lite trades some capability for better availability.
_FALLBACK_MODELS = ["gemini-2.5-flash-lite"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact_key(text, api_key):
    if api_key and api_key in str(text):
        return str(text).replace(api_key, "[REDACTED]")
    return str(text)


#: Gemini hands back grounding sources as redirect wrappers on this host rather
#: than as the source URLs themselves.
_GROUNDING_REDIRECT_HOST = "vertexaisearch.cloud.google.com"


def resolve_grounding_urls(chunks, timeout=20, cache=None):
    """Turn Gemini grounding chunks into real source URLs.

    ``chunks`` is what :func:`ci_core.llm.streaming.accumulate_gemini` collected:
    ``[{"uri", "title"}, ...]`` where each ``uri`` is a
    ``vertexaisearch.cloud.google.com/grounding-api-redirect/...`` wrapper. Each
    one has to be followed to learn the URL it stands for.

    Why the wrapper must never be stored: those redirect links expire after
    roughly 30 days. Putting one in the drift index or handing it to Wayback
    preserves a link that will be dead by the time anyone follows it, which is
    worse than having no URL — it looks like a citation and isn't one. So a chunk
    whose redirect cannot be resolved is **dropped**, not downgraded: ``title``
    carries only a bare domain ("epa.gov"), and a homepage is not the source the
    claim rests on. No URL beats a wrong one.

    Resolution is deliberately off the streaming path — it is one HTTP request
    per chunk, and doing it inside the model call would spend the call's timeout
    budget on it. ``cache`` (a dict) is shared across calls in a run so the same
    redirect is followed once.

    Returns the resolved source URLs in first-seen order.
    """
    from ...http import DEFAULT_HEADERS

    if cache is None:
        cache = {}
    resolved = []
    for chunk in chunks or []:
        uri = chunk.get("uri")
        if not isinstance(uri, str) or not uri:
            continue
        if uri not in cache:
            cache[uri] = _follow_redirect(uri, timeout, DEFAULT_HEADERS)
        final = cache[uri]
        if final and final not in resolved:
            resolved.append(final)
    dropped = len([c for c in chunks or [] if c.get("uri")]) - len(resolved)
    if dropped > 0:
        log.debug(
            f"Gemini grounding: {len(resolved)} source URL(s) resolved, "
            f"{dropped} redirect(s) unresolvable and dropped."
        )
    return resolved


def _follow_redirect(uri, timeout, headers):
    """Return the final URL behind a grounding redirect, or None.

    ``None`` for anything that did not land on a real source: a network failure,
    an error status, or a URL still on the redirect host (meaning the redirect
    did not actually go anywhere). Some of these legitimately fail — a redirect
    can land on a bot-detection interstitial — and that is a drop, not an error
    worth failing the run over.
    """
    try:
        resp = requests.get(uri, timeout=timeout, allow_redirects=True, headers=headers)
    except requests.RequestException as e:
        log.debug(f"Gemini grounding redirect failed ({type(e).__name__}): {uri[:80]}")
        return None
    if resp.status_code >= 400:
        log.debug(f"Gemini grounding redirect returned {resp.status_code}: {uri[:80]}")
        return None
    final = resp.url or ""
    if not final or _GROUNDING_REDIRECT_HOST in final:
        return None
    return final


def _is_capacity_error(exc):
    """Return True if the exception is a 503 capacity/availability error."""
    msg = str(exc)
    return "503" in msg and (
        "UNAVAILABLE" in msg or "high demand" in msg or "Service Unavailable" in msg
    )


def _resolve_model(model_arg, provider_config):
    """Resolve the model name from explicit argument or provider_config."""
    return model_arg or (provider_config or {}).get("model") or DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def call(
    system_prompt,
    user_prompt,
    api_key,
    retry=True,
    retry_delay=10,
    model=None,
    provider_config=None,
):
    cfg = provider_config or {}
    provider = cfg.get("provider", "ai_studio")

    if provider == "vertex_ai":
        return _call_with_fallback(
            system_prompt,
            user_prompt,
            api_key,
            retry=retry,
            retry_delay=retry_delay,
            model=model,
            cfg=cfg,
            backend="vertex",
        )
    else:
        return _call_with_fallback(
            system_prompt,
            user_prompt,
            api_key,
            retry=retry,
            retry_delay=retry_delay,
            model=model,
            cfg=cfg,
            backend="aistudio",
        )


# ---------------------------------------------------------------------------
# Fallback-chain orchestration (shared by both backends)
# ---------------------------------------------------------------------------


def _call_with_fallback(
    system_prompt,
    user_prompt,
    api_key,
    retry,
    retry_delay,
    model,
    cfg,
    backend,
):
    requested_model = _resolve_model(model, cfg)
    models_to_try = [requested_model] + [
        m for m in _FALLBACK_MODELS if m != requested_model
    ]

    result = None
    for attempt_model in models_to_try:
        if backend == "vertex":
            result = _call_vertex(
                system_prompt,
                user_prompt,
                cfg,
                model=attempt_model,
                retry=retry,
                retry_delay=retry_delay,
            )
        else:
            result = _call_aistudio(
                system_prompt,
                user_prompt,
                api_key,
                model=attempt_model,
                retry=retry,
                retry_delay=retry_delay,
                provider_config=cfg,
            )

        if not result.get("failed"):
            if attempt_model != requested_model:
                result["fallback_from"] = requested_model
                log.warning(
                    f"Gemini fact-check used FALLBACK model {attempt_model!r} "
                    f"because {requested_model!r} was unavailable (capacity). "
                    f"Search grounding and full capability may be reduced."
                )
            return result

        if (
            _is_capacity_error(result.get("error", ""))
            and attempt_model != models_to_try[-1]
        ):
            log.warning(
                f"Gemini {attempt_model} unavailable (capacity). "
                f"Trying next fallback model."
            )
            continue

        return result  # non-capacity failure or exhausted fallbacks

    return result


# ---------------------------------------------------------------------------
# AI Studio backend
# ---------------------------------------------------------------------------


def _aistudio_url(model, api_key):
    # streamGenerateContent + alt=sse → Server-Sent Events instead of one buffered body.
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:streamGenerateContent?alt=sse&key={api_key}"
    )


def _call_aistudio(
    system_prompt,
    user_prompt,
    api_key,
    model,
    retry=True,
    retry_delay=10,
    provider_config=None,
):
    url = _aistudio_url(model, api_key)
    grounding_tool = {"google_search": {}}
    return _execute_request(
        system_prompt,
        user_prompt,
        url=url,
        headers={"Content-Type": "application/json"},
        grounding_tool=grounding_tool,
        model=model,
        api_key=api_key,
        retry=retry,
        retry_delay=retry_delay,
        provider_config=provider_config,
    )


# ---------------------------------------------------------------------------
# Vertex AI backend
# ---------------------------------------------------------------------------


def _vertex_url(model, cfg):
    project = cfg.get("project", "")
    location = cfg.get("location", "us-central1")
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse"
    )


def _get_vertex_token(cfg):
    """Return a short-lived Bearer token for Vertex AI.

    Uses a service-account JSON file if ``credentials_file`` is set in cfg,
    otherwise falls back to Application Default Credentials (gcloud auth,
    Workload Identity, etc.).

    Raises ImportError with an actionable message if google-auth is absent.
    """
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError:
        raise ImportError(
            "The 'google-auth' package is required for Vertex AI.\n"
            "Install it with:  pip install google-auth\n"
            "Or install all optional dependencies:  pip install -r requirements-optional.txt"
        ) from None

    credentials_file = cfg.get("credentials_file")
    if credentials_file:
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    else:
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token


def _call_vertex(system_prompt, user_prompt, cfg, model, retry=True, retry_delay=10):
    url = _vertex_url(model, cfg)
    try:
        token = _get_vertex_token(cfg)
    except Exception as e:
        return {
            "failed": True,
            "error": str(e),
            "raw": None,
            "model": model,
            "tokens": {},
            "grounding_available": False,
            "elapsed_seconds": 0,
        }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    # Both Vertex AI and AI Studio now use the same google_search tool name.
    grounding_tool = {"google_search": {}}
    return _execute_request(
        system_prompt,
        user_prompt,
        url=url,
        headers=headers,
        grounding_tool=grounding_tool,
        model=model,
        api_key=None,  # no key to redact in Vertex error messages
        retry=retry,
        retry_delay=retry_delay,
        provider_config=cfg,
    )


# ---------------------------------------------------------------------------
# Shared HTTP execution (both backends use the same response handling)
# ---------------------------------------------------------------------------


def _execute_request(
    system_prompt,
    user_prompt,
    url,
    headers,
    grounding_tool,
    model,
    api_key,
    retry,
    retry_delay,
    provider_config=None,
):
    """Send the prompt to Gemini and parse the response.

    Tries the grounded payload first (enables live search).  If that fails
    (e.g. the model doesn't support grounding, or a 400 is returned), falls
    back to a plain payload without grounding.
    """
    _grounded_gen_config = {"temperature": 0.2}
    # responseMimeType is incompatible with grounding — used only on plain payload.
    _plain_gen_config = {"temperature": 0.2, "responseMimeType": "application/json"}
    # A response schema is a stronger form of the same constraint and inherits the
    # same limitation: Gemini rejects it alongside google_search with HTTP 400
    # ("Tool use with a response mime type: 'application/json' is unsupported",
    # verified live 2026-08-12). So the grounded fact-check keeps asking for JSON
    # in the prompt, and only the ungrounded fallback gets it enforced.
    _schema = (provider_config or {}).get("response_schema")
    if _schema:
        _converted = schema_format.gemini_response_schema(_schema)
        if _converted:
            _plain_gen_config.update(_converted)

    # Optional thinking budget from provider_config (e.g. thinking_budget: 8192).
    # gemini-2.5-flash already uses dynamic thinking by default; setting an explicit
    # budget controls how many tokens the model can spend on internal reasoning.
    # Set to 0 in config to disable thinking entirely (faster/cheaper for simple tasks).
    thinking_budget = (provider_config or {}).get("thinking_budget")
    if thinking_budget is not None:
        # includeThoughts: true ensures the response includes thought parts so our
        # filtering logic can strip them. Without it the API may omit them on some models.
        _grounded_gen_config["thinkingConfig"] = {
            "thinkingBudget": thinking_budget,
            "includeThoughts": True,
        }
        _plain_gen_config["thinkingConfig"] = {
            "thinkingBudget": thinking_budget,
            "includeThoughts": True,
        }

    payload_grounded = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "tools": [grounding_tool],
        "generationConfig": _grounded_gen_config,
    }
    payload_plain = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": _plain_gen_config,
    }

    # Streaming socket timeout = inter-token read-gap (small constant). The big
    # sliding-scale timeout_seconds survives only as the pipeline's wall-clock backstop.
    timeout = streaming.stream_timeout(provider_config, _READ_TIMEOUT)

    session = requests.Session()
    t0 = time.monotonic()

    def _post(payload):
        resp = session.post(
            url, headers=headers, json=payload, stream=True, timeout=timeout
        )
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(
                f"Gemini {model} HTTP {resp.status_code}. "
                f"Waiting {retry_delay}s before retry."
            )
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(
                url, headers=headers, json=payload, stream=True, timeout=timeout
            )
        resp.raise_for_status()
        return streaming.accumulate_gemini(resp)

    grounding_available = True
    try:
        assembled = _post(payload_grounded)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        # A timeout/connection failure means the endpoint is slow or unreachable, not
        # that grounding is unsupported. Retrying without grounding would just burn the
        # same budget again and risk the pipeline force-cancelling the whole pass (which
        # discards any result). Fail fast and cleanly instead.
        elapsed = round(time.monotonic() - t0, 2)
        safe_err = _redact_key(e, api_key)
        # This adapter is called once per (model, domain) pair but is never told
        # which domain it's running — it must not name one here. pipeline.py's
        # [CALIBRATION] log line reports the real domain from its own (model,
        # domain) bookkeeping.
        log.error(
            f"Gemini {model} timed out after {elapsed}s on the grounded call. "
            f"Not retrying without grounding. This is the inter-token read-gap "
            f"timeout (default {_READ_TIMEOUT}s, covers time-to-first-byte while "
            f"the model searches/thinks) — raise stream_read_timeout for gemini "
            f"in user.yaml if this repeats, not timeout_seconds."
        )
        session.close()
        return {
            "failed": True,
            "error": safe_err,
            "raw": None,
            "model": model,
            "tokens": {},
            "grounding_available": False,
            "elapsed_seconds": elapsed,
        }
    except requests.HTTPError as e:
        body = _redact_key(redact.capture_error_body(e), api_key)
        safe_err = _redact_key(e, api_key)
        log.warning(
            f"Gemini {model} with search grounding failed: {safe_err} | {body}. "
            f"Retrying without grounding."
        )
        grounding_available = False
        try:
            assembled = _post(payload_plain)
        except requests.HTTPError as e2:
            elapsed = round(time.monotonic() - t0, 2)
            body2 = _redact_key(redact.capture_error_body(e2), api_key)
            safe_err2 = _redact_key(e2, api_key)
            log.error(
                f"Gemini {model} call failed entirely after {elapsed}s: {safe_err2} | {body2}"
            )
            session.close()
            return {
                "failed": True,
                "error": safe_err2,
                "error_body": body2,
                "raw": None,
                "model": model,
                "tokens": {},
                "grounding_available": False,
                "elapsed_seconds": elapsed,
            }
        except Exception as e2:
            elapsed = round(time.monotonic() - t0, 2)
            safe_err2 = _redact_key(e2, api_key)
            error_body2 = _redact_key(redact.capture_error_body(e2), api_key)
            log.error(
                f"Gemini {model} call failed entirely after {elapsed}s: {safe_err2}"
            )
            session.close()
            return {
                "failed": True,
                "error": safe_err2,
                "error_body": error_body2,
                "raw": None,
                "model": model,
                "tokens": {},
                "grounding_available": False,
                "elapsed_seconds": elapsed,
            }
    except Exception as e:
        safe_err = _redact_key(e, api_key)
        log.warning(
            f"Gemini {model} with search grounding failed: {safe_err}. "
            f"Retrying without grounding."
        )
        grounding_available = False
        try:
            assembled = _post(payload_plain)
        except Exception as e2:
            elapsed = round(time.monotonic() - t0, 2)
            safe_err2 = _redact_key(e2, api_key)
            error_body2 = _redact_key(redact.capture_error_body(e2), api_key)
            log.error(
                f"Gemini {model} call failed entirely after {elapsed}s: {safe_err2}"
            )
            session.close()
            return {
                "failed": True,
                "error": safe_err2,
                "error_body": error_body2,
                "raw": None,
                "model": model,
                "tokens": {},
                "grounding_available": False,
                "elapsed_seconds": elapsed,
            }

    elapsed = round(time.monotonic() - t0, 2)
    session.close()

    # accumulate_gemini already concatenated the streamed text parts in order,
    # skipping parts marked ``thought: true`` (internal reasoning that would
    # otherwise produce malformed JSON), and kept the final cumulative usageMetadata.
    text = assembled["content"]
    usage = assembled["usage"]
    finish_reason = assembled.get("finish_reason")
    if not text:
        return {
            "failed": True,
            "error": "No candidates in Gemini response",
            "raw": None,
            "model": model,
            "tokens": {},
            "grounding_available": grounding_available,
            "elapsed_seconds": elapsed,
        }

    # Robust parse: handles fences and prose/reasoning preambles that survive the
    # thought-part filtering above. Shared with the other review adapters.
    # Falls back to truncation salvage, because Gemini hits the output-token
    # ceiling routinely — thinking tokens draw from the same budget as the
    # answer, so a large thinking_budget can cut the JSON off mid-array even
    # when finishReason comes back as STOP.
    parsed, truncated = _extract_json_with_salvage(text)
    if parsed is None:
        # finishReason=MAX_TOKENS means generation was cut off mid-output (often
        # because the thinking budget consumed most of the token budget before
        # any answer text was produced) — a genuinely truncated payload, not the
        # model returning malformed content. Surface that distinction so it's
        # diagnosable without needing the raw text.
        if finish_reason == "MAX_TOKENS":
            error_msg = (
                "Malformed JSON response (truncated: finishReason=MAX_TOKENS — "
                "output was cut off before valid JSON completed; raise "
                "max_output_tokens or lower thinking_budget)"
            )
        elif finish_reason and finish_reason != "STOP":
            error_msg = f"Malformed JSON response (finishReason={finish_reason})"
        else:
            error_msg = "Malformed JSON response"
        log.warning(
            f"Gemini {model} returned non-JSON content after {elapsed}s "
            f"(finishReason={finish_reason})"
        )
        return {
            "failed": True,
            "error": error_msg,
            "raw": text,
            "model": model,
            "tokens": normalize_tokens(usage),
            "grounding_available": grounding_available,
            "elapsed_seconds": elapsed,
            "finish_reason": finish_reason,
        }

    if truncated:
        log.warning(
            f"Gemini {model} response was truncated (finishReason={finish_reason}, "
            f"{usage.get('candidatesTokenCount', '?')} candidate + "
            f"{usage.get('thoughtsTokenCount', '?')} thinking tokens) after "
            f"{elapsed}s; salvaged the complete elements, discarded the rest."
        )
    else:
        log.debug(
            f"Gemini {model} call succeeded in {elapsed}s "
            f"(grounding={grounding_available})"
        )
    result = {
        "failed": False,
        "raw": text,
        "data": parsed,
        "model": model,
        "tokens": normalize_tokens(usage),
        "grounding_available": grounding_available,
        # Raw redirect wrappers, not source URLs. The caller resolves them with
        # resolve_grounding_urls() once per run, off the timed call path.
        "grounding_chunks": assembled.get("grounding_chunks") or [],
        "elapsed_seconds": elapsed,
    }
    if truncated:
        # Not "failed" — the recovered findings are real — but flagged so
        # downstream reporting can distinguish this from a clean response.
        result["truncated"] = True
        result["finish_reason"] = finish_reason
    return result
