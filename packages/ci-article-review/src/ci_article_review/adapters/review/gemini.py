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

import json
import time
import logging
import requests

from . import streaming
from .json_utils import extract_json as _extract_json

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
            system_prompt, user_prompt, api_key,
            retry=retry, retry_delay=retry_delay,
            model=model, cfg=cfg, backend="vertex",
        )
    else:
        return _call_with_fallback(
            system_prompt, user_prompt, api_key,
            retry=retry, retry_delay=retry_delay,
            model=model, cfg=cfg, backend="aistudio",
        )


# ---------------------------------------------------------------------------
# Fallback-chain orchestration (shared by both backends)
# ---------------------------------------------------------------------------

def _call_with_fallback(
    system_prompt, user_prompt, api_key,
    retry, retry_delay, model, cfg, backend,
):
    requested_model = _resolve_model(model, cfg)
    models_to_try = [requested_model] + [
        m for m in _FALLBACK_MODELS if m != requested_model
    ]

    result = None
    for attempt_model in models_to_try:
        if backend == "vertex":
            result = _call_vertex(
                system_prompt, user_prompt, cfg,
                model=attempt_model, retry=retry, retry_delay=retry_delay,
            )
        else:
            result = _call_aistudio(
                system_prompt, user_prompt, api_key,
                model=attempt_model, retry=retry, retry_delay=retry_delay,
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

        if _is_capacity_error(result.get("error", "")) and attempt_model != models_to_try[-1]:
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


def _call_aistudio(system_prompt, user_prompt, api_key, model, retry=True, retry_delay=10, provider_config=None):
    url = _aistudio_url(model, api_key)
    grounding_tool = {"google_search": {}}
    return _execute_request(
        system_prompt, user_prompt,
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
        system_prompt, user_prompt,
        url=url,
        headers=headers,
        grounding_tool=grounding_tool,
        model=model,
        api_key=None,       # no key to redact in Vertex error messages
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

    # Optional thinking budget from provider_config (e.g. thinking_budget: 8192).
    # gemini-2.5-flash already uses dynamic thinking by default; setting an explicit
    # budget controls how many tokens the model can spend on internal reasoning.
    # Set to 0 in config to disable thinking entirely (faster/cheaper for simple tasks).
    thinking_budget = (provider_config or {}).get("thinking_budget")
    if thinking_budget is not None:
        # includeThoughts: true ensures the response includes thought parts so our
        # filtering logic can strip them. Without it the API may omit them on some models.
        _grounded_gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget, "includeThoughts": True}
        _plain_gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget, "includeThoughts": True}

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
        resp = session.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(
                f"Gemini {model} HTTP {resp.status_code}. "
                f"Waiting {retry_delay}s before retry."
            )
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
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
        log.error(
            f"Gemini {model} timed out after {elapsed}s on the grounded fact-check. "
            f"Not retrying without grounding. For long articles, raise the gemini "
            f"timeout_seconds in user.yaml (and task_timeout_seconds to match)."
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
        body = _redact_key(e.response.text[:400] if e.response is not None else "", api_key)
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
            body2 = _redact_key(e2.response.text[:400] if e2.response is not None else "", api_key)
            safe_err2 = _redact_key(e2, api_key)
            log.error(f"Gemini {model} call failed entirely after {elapsed}s: {safe_err2} | {body2}")
            session.close()
            return {
                "failed": True,
                "error": safe_err2,
                "raw": None,
                "model": model,
                "tokens": {},
                "grounding_available": False,
                "elapsed_seconds": elapsed,
            }
        except Exception as e2:
            elapsed = round(time.monotonic() - t0, 2)
            safe_err2 = _redact_key(e2, api_key)
            log.error(f"Gemini {model} call failed entirely after {elapsed}s: {safe_err2}")
            session.close()
            return {
                "failed": True,
                "error": safe_err2,
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
            log.error(f"Gemini {model} call failed entirely after {elapsed}s: {safe_err2}")
            session.close()
            return {
                "failed": True,
                "error": safe_err2,
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
    parsed = _extract_json(text)
    if parsed is None:
        log.warning(f"Gemini {model} returned non-JSON content after {elapsed}s")
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": text,
            "model": model,
            "tokens": usage,
            "grounding_available": grounding_available,
            "elapsed_seconds": elapsed,
        }

    log.debug(
        f"Gemini {model} call succeeded in {elapsed}s "
        f"(grounding={grounding_available})"
    )
    return {
        "failed": False,
        "data": parsed,
        "model": model,
        "tokens": {
            "prompt": usage.get("promptTokenCount"),
            "completion": usage.get("candidatesTokenCount"),
        },
        "grounding_available": grounding_available,
        "elapsed_seconds": elapsed,
    }
