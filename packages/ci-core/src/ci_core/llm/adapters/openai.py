"""OpenAI voice/completeness adapter.

Supports three providers:
  - openai       (default) — api.openai.com Responses API, Bearer-token auth.
  - openai_search          — api.openai.com Responses API with web_search_preview tool.
                             Set ``web_search: true`` in the model config to enable.
                             Falls back to the standard (non-search) path on error.
  - azure                  — Azure OpenAI Service, api-key header auth, Chat Completions.

Provider is selected via the ``provider_config`` dict passed by the pipeline.
Existing user.yaml files that specify a plain model string continue to work
unchanged.

The openai.com path (default and web_search) both use the Responses API
(``/v1/responses``), not Chat Completions. Reasoning models (reasoning_effort
set) request ``reasoning.summary`` so the stream carries
``response.reasoning_summary_text.*`` events during the silent "thinking"
phase — see streaming.py and the read-gap timeout discussion in
ci_core/llm/streaming.py and configs/presets.yaml. Azure still uses Chat
Completions (``_execute_request`` / ``OPENAI_API_URL``) since only the
openai.com path was migrated.

Web search config example (configs/user.yaml)::

    models:
      openai:
        model: gpt-4o
        web_search: true    # enables Responses API with live web search

Azure OpenAI configuration example (configs/user.yaml)::

    models:
      openai:
        provider: azure
        model: gpt-4o           # informational only; deployment name drives the request
        endpoint: https://my-resource.openai.azure.com
        deployment: my-gpt4o-deployment
        api_version: "2024-02-01"   # optional; defaults to 2024-02-01

Azure note: provisioned throughput deployments do not 503; the fallback chain
is bypassed for Azure and the configured deployment is used directly.
"""

import time
import logging
import requests

from .. import streaming
from ..json_utils import extract_json_with_salvage as _extract_json_with_salvage
from ..tokens import normalize_tokens
from ... import redact

DEFAULT_MODEL = "gpt-5.4"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

# Inter-token read-gap timeout (seconds) for the streaming socket. Constant, not
# derived from the sliding-scale timeout_seconds — see ci_core/llm/streaming.py.
_READ_TIMEOUT = streaming.DEFAULT_READ_TIMEOUT

# Fallback chain tried in order when primary model returns a capacity error (503).
# gpt-5.4-mini has the same API surface but reduced capability and reasoning depth.
_FALLBACK_MODELS = ["gpt-5.4-mini"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_capacity_error(exc):
    """Return True if the exception represents a 503 capacity/availability error."""
    return "503" in str(exc)


def _resolve_model(model_arg, cfg):
    return model_arg or cfg.get("model") or DEFAULT_MODEL


def _parse_json_maybe_fenced(content):
    """Parse ``content`` as JSON, tolerating fences, preambles, and truncation.

    The Responses API has no ``response_format: json_object``, so JSON is
    requested via the system prompt / instructions and occasionally comes back
    wrapped in a ```` ```json ... ``` ```` fence or behind a reasoning preamble.
    Delegates to the shared extractor — a superset of the leading-fence handling
    this was doing inline — and falls back to truncation salvage for a response
    cut off at the output-token ceiling.

    Returns ``(parsed, None, truncated)`` on success or ``(None, content,
    False)`` on failure (the original text, for the caller's ``raw`` field).
    """
    parsed, truncated = _extract_json_with_salvage(content)
    if parsed is None:
        return None, content, False
    return parsed, None, truncated


def _log_outcome(label, truncated, elapsed, usage):
    """Log a successful call, distinguishing a salvaged truncation from a clean one."""
    if truncated:
        log.warning(
            f"{label} response was truncated (likely hit the output-token ceiling "
            f"— {usage.get('output_tokens') or usage.get('completion_tokens', '?')} "
            f"output tokens) after {elapsed}s; salvaged the complete elements, "
            f"discarded the rest."
        )
    else:
        log.debug(f"{label} call succeeded in {elapsed}s")


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
    provider = cfg.get("provider", "openai")

    # Streaming: the socket timeout is the inter-token read-gap, not the whole
    # generation time, so it stays small and constant regardless of how long the
    # model reasons. The big sliding-scale timeout_seconds is no longer a socket
    # timeout — it survives only as the pipeline's per-task wall-clock backstop.
    timeout = streaming.stream_timeout(cfg, _READ_TIMEOUT)

    if provider == "azure":
        return _call_azure(
            system_prompt,
            user_prompt,
            api_key,
            cfg,
            retry=retry,
            retry_delay=retry_delay,
            timeout=timeout,
        )

    # web_search: true → try Responses API with live search, fall back if unavailable
    if cfg.get("web_search"):
        requested_model = _resolve_model(model, cfg)
        result = _call_with_web_search(
            system_prompt,
            user_prompt,
            api_key,
            model=requested_model,
            retry=retry,
            retry_delay=retry_delay,
            timeout=timeout,
        )
        if not result.get("failed"):
            return result
        log.warning(
            f"OpenAI web search call failed ({result.get('error')}); "
            "falling back to standard chat completions."
        )
        # Fall through to standard path below

    # --- openai.com path with fallback chain ---
    requested_model = _resolve_model(model, cfg)
    reasoning_effort = cfg.get("reasoning_effort")
    models_to_try = [requested_model] + [
        m for m in _FALLBACK_MODELS if m != requested_model
    ]

    result = None
    for attempt_model in models_to_try:
        result = _call_openai(
            system_prompt,
            user_prompt,
            api_key,
            model=attempt_model,
            retry=retry,
            retry_delay=retry_delay,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        if not result.get("failed"):
            if attempt_model != requested_model:
                result["fallback_from"] = requested_model
                log.warning(
                    f"OpenAI used FALLBACK model {attempt_model!r} because "
                    f"{requested_model!r} was unavailable (capacity). "
                    f"Review quality may be reduced."
                )
            return result
        if (
            _is_capacity_error(result.get("error", ""))
            and attempt_model != models_to_try[-1]
        ):
            log.warning(
                f"OpenAI {attempt_model} unavailable (capacity). Trying next fallback model."
            )
            continue
        return result

    return result  # exhausted fallbacks


# ---------------------------------------------------------------------------
# OpenAI Responses API (web search)
# ---------------------------------------------------------------------------


def _call_with_web_search(
    system_prompt, user_prompt, api_key, model, retry, retry_delay, timeout=None
):
    """Call via the OpenAI Responses API with the web_search_preview tool.

    The Responses API has a different payload and response shape from chat
    completions.  ``instructions`` carries the system prompt; ``input`` carries
    the user message.  Streamed output arrives as typed events
    (``response.output_text.delta``); :func:`streaming.accumulate_openai_responses`
    reassembles the text and final usage.  The endpoint does not support
    ``response_format: json_object``, so JSON is requested via the system prompt.
    """
    if timeout is None:
        timeout = streaming.stream_timeout(None, _READ_TIMEOUT)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "tools": [{"type": "web_search_preview"}],
        "temperature": 0.2,
        "stream": True,
    }

    session = requests.Session()
    t0 = time.monotonic()

    def _post():
        resp = session.post(
            OPENAI_RESPONSES_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
        )
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(
                f"OpenAI Responses API {model} HTTP {resp.status_code}. "
                f"Waiting {retry_delay}s before retry."
            )
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(
                OPENAI_RESPONSES_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout,
            )
        resp.raise_for_status()
        return resp

    try:
        resp = _post()
        assembled = streaming.accumulate_openai_responses(resp)
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        error_body = redact.capture_error_body(e)
        if error_body:
            log.warning(
                f"OpenAI Responses API {model} call failed after {elapsed}s: {e} | {error_body}"
            )
        else:
            log.warning(
                f"OpenAI Responses API {model} call failed after {elapsed}s: {e}"
            )
        session.close()
        return {
            "failed": True,
            "error": str(e),
            "error_body": error_body,
            "raw": None,
            "model": model,
            "tokens": {},
            "elapsed_seconds": elapsed,
            "grounding_available": False,
        }
    finally:
        session.close()

    elapsed = round(time.monotonic() - t0, 2)
    usage = assembled["usage"]
    content = assembled["content"]

    # Parse JSON — Responses API output may be wrapped in code fences
    parsed, raw_on_fail, truncated = _parse_json_maybe_fenced(content)
    if parsed is None:
        log.warning(
            f"OpenAI (web search) {model} returned non-JSON content after {elapsed}s"
        )
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": raw_on_fail,
            "model": model,
            "tokens": normalize_tokens(usage),
            "elapsed_seconds": elapsed,
            "grounding_available": True,
        }

    _log_outcome(f"OpenAI (web search) {model}", truncated, elapsed, usage)
    result = {
        "failed": False,
        "raw": content,
        "data": parsed,
        "model": f"{model}+search",
        "tokens": normalize_tokens(usage),
        "elapsed_seconds": elapsed,
        "grounding_available": True,
    }
    if truncated:
        result["truncated"] = True
    return result


# ---------------------------------------------------------------------------
# openai.com backend (Responses API)
# ---------------------------------------------------------------------------


def _call_openai(
    system_prompt,
    user_prompt,
    api_key,
    model,
    retry=True,
    retry_delay=10,
    reasoning_effort=None,
    timeout=None,
):
    """Call the openai.com Responses API — the primary (non-search) path.

    Migrated from Chat Completions: reasoning models sent zero bytes over the
    wire during their "thinking" phase there, forcing the read-gap timeout up
    to 200-300s for high/xhigh effort (see configs/presets.yaml). Requesting
    ``reasoning.summary`` on the Responses API streams
    ``response.reasoning_summary_text.*`` events during that same phase,
    keeping the socket active.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _build_payload(with_reasoning):
        p = {
            "model": model,
            "instructions": system_prompt,
            "input": user_prompt,
            "stream": True,
        }
        if with_reasoning and reasoning_effort:
            # effort: "none" | "low" | "medium" | "high" | "max"; supported on
            # o-series and reasoning-capable models. summary: "auto" streams
            # reasoning_summary_text events. temperature is incompatible with
            # reasoning mode, same as it was under Chat Completions.
            p["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        else:
            # No response_format: json_object on the Responses API — JSON is
            # requested via the system prompt / instructions instead (same
            # constraint _call_with_web_search already handles).
            p["temperature"] = 0.2
        return p

    result = _execute_openai_responses(
        headers,
        _build_payload(True),
        model,
        url=OPENAI_RESPONSES_URL,
        retry=retry,
        retry_delay=retry_delay,
        timeout=timeout,
    )

    # If the model rejected reasoning, retry without it so the pass still runs.
    # This is a MISCONFIGURATION: the preset specifies a reasoning model (o4-mini, o3)
    # but user.yaml overrides it with a non-reasoning model. Fix by not overriding the
    # model in user.yaml for balanced+ presets, or switching to a standard/economy preset.
    if (
        reasoning_effort
        and result.get("failed")
        and _is_reasoning_param_error(result.get("error_body", ""))
    ):
        msg = (
            f"[MISCONFIGURATION] OpenAI {model} rejected reasoning_effort={reasoning_effort!r}. "
            f"Reasoning is only supported on o-series models (o4-mini, o3, etc.). "
            f"If using a balanced+ preset, do not override the OpenAI model in user.yaml "
            f"with a non-reasoning variant. Retrying without reasoning — output may be degraded."
        )
        log.error(msg)
        retry_result = _execute_openai_responses(
            headers,
            _build_payload(False),
            model,
            url=OPENAI_RESPONSES_URL,
            retry=retry,
            retry_delay=retry_delay,
            timeout=timeout,
        )
        retry_result["misconfiguration_warning"] = msg
        return retry_result

    return result


def _is_reasoning_param_error(body_text):
    """Return True if a 400 body indicates an unsupported reasoning parameter."""
    lower = body_text.lower()
    return (
        "unknown_parameter" in lower or "unsupported parameter" in lower
    ) and "reasoning" in lower


# ---------------------------------------------------------------------------
# Azure OpenAI backend
# ---------------------------------------------------------------------------


def _call_azure(
    system_prompt, user_prompt, api_key, cfg, retry=True, retry_delay=10, timeout=None
):
    endpoint = cfg.get("endpoint", "").rstrip("/")
    deployment = cfg.get("deployment", "")
    api_version = cfg.get("api_version", "2024-02-01")
    model_name = cfg.get("model", deployment or DEFAULT_MODEL)

    if not endpoint or not deployment:
        return {
            "failed": True,
            "error": (
                "Azure OpenAI requires 'endpoint' and 'deployment' in the model config. "
                "See configs/user.example.yaml for the extended provider format."
            ),
            "raw": None,
            "model": model_name,
            "tokens": {},
            "elapsed_seconds": 0,
        }

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": deployment,  # Azure accepts this; ignored by the service but harmless
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = _execute_request(
        headers,
        payload,
        model_name,
        api_key=None,
        url=url,
        retry=retry,
        retry_delay=retry_delay,
        timeout=timeout,
    )
    # Tag so callers know which endpoint was used
    result["provider"] = "azure"
    return result


# ---------------------------------------------------------------------------
# openai.com Responses API execution (primary + fallback-without-reasoning)
# ---------------------------------------------------------------------------


def _execute_openai_responses(
    headers, payload, model, url, retry, retry_delay, timeout=None
):
    """POST to the Responses API, stream via ``accumulate_openai_responses``,
    and parse the assembled text as JSON. Mirrors ``_execute_request``'s
    retry/error shape so ``_call_openai``'s misconfiguration-retry logic
    (``error_body`` + ``_is_reasoning_param_error``) works unchanged.
    """
    if timeout is None:
        timeout = streaming.stream_timeout(None, _READ_TIMEOUT)
    session = requests.Session()
    t0 = time.monotonic()

    def _post():
        resp = session.post(
            url, headers=headers, json=payload, stream=True, timeout=timeout
        )
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(
                f"OpenAI {model} HTTP {resp.status_code}. Waiting {retry_delay}s before retry."
            )
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(
                url, headers=headers, json=payload, stream=True, timeout=timeout
            )
        resp.raise_for_status()
        return resp

    try:
        resp = _post()
        assembled = streaming.accumulate_openai_responses(resp)
    except requests.HTTPError as e:
        elapsed = round(time.monotonic() - t0, 2)
        body = redact.capture_error_body(e)
        log.error(f"OpenAI {model} call failed after {elapsed}s: {e} | {body}")
        return {
            "failed": True,
            "error": str(e),
            "error_body": body,
            "raw": None,
            "model": model,
            "tokens": {},
            "elapsed_seconds": elapsed,
        }
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        log.error(f"OpenAI {model} call failed after {elapsed}s: {e}")
        return {
            "failed": True,
            "error": str(e),
            "error_body": "",
            "raw": None,
            "model": model,
            "tokens": {},
            "elapsed_seconds": elapsed,
        }
    finally:
        session.close()

    elapsed = round(time.monotonic() - t0, 2)
    content = assembled["content"]
    usage = assembled["usage"]
    tokens = normalize_tokens(usage)

    parsed, raw_on_fail, truncated = _parse_json_maybe_fenced(content)
    if parsed is None:
        log.warning(f"OpenAI {model} returned non-JSON content after {elapsed}s")
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": raw_on_fail,
            "model": model,
            "tokens": tokens,
            "elapsed_seconds": elapsed,
        }

    _log_outcome(f"OpenAI {model}", truncated, elapsed, usage)
    result = {
        "failed": False,
        "raw": content,
        "data": parsed,
        "model": model,
        "tokens": tokens,
        "elapsed_seconds": elapsed,
    }
    if truncated:
        result["truncated"] = True
    return result


# ---------------------------------------------------------------------------
# Shared HTTP execution (Azure — Chat Completions)
# ---------------------------------------------------------------------------


def _execute_request(
    headers, payload, model, api_key, url, retry, retry_delay, timeout=None
):
    if timeout is None:
        timeout = streaming.stream_timeout(None, _READ_TIMEOUT)
    session = requests.Session()
    t0 = time.monotonic()

    def _post():
        resp = session.post(
            url, headers=headers, json=payload, stream=True, timeout=timeout
        )
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(
                f"OpenAI {model} HTTP {resp.status_code}. Waiting {retry_delay}s before retry."
            )
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(
                url, headers=headers, json=payload, stream=True, timeout=timeout
            )
        resp.raise_for_status()
        return resp

    try:
        resp = _post()
        assembled = streaming.accumulate_chat_completions(resp)
    except requests.HTTPError as e:
        elapsed = round(time.monotonic() - t0, 2)
        body = redact.capture_error_body(e)
        log.error(f"OpenAI {model} call failed after {elapsed}s: {e} | {body}")
        return {
            "failed": True,
            "error": str(e),
            "error_body": body,
            "raw": None,
            "model": model,
            "tokens": {},
            "elapsed_seconds": elapsed,
        }
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        log.error(f"OpenAI {model} call failed after {elapsed}s: {e}")
        return {
            "failed": True,
            "error": str(e),
            "error_body": "",
            "raw": None,
            "model": model,
            "tokens": {},
            "elapsed_seconds": elapsed,
        }
    finally:
        session.close()

    elapsed = round(time.monotonic() - t0, 2)
    content = assembled["content"]
    usage = assembled["usage"]

    parsed, truncated = _extract_json_with_salvage(content)
    if parsed is None:
        log.warning(f"OpenAI {model} returned non-JSON content after {elapsed}s")
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": content,
            "model": model,
            "tokens": normalize_tokens(usage),
            "elapsed_seconds": elapsed,
        }

    _log_outcome(f"OpenAI {model}", truncated, elapsed, usage)
    result = {
        "failed": False,
        "raw": content,
        "data": parsed,
        "model": model,
        "tokens": normalize_tokens(usage),
        "elapsed_seconds": elapsed,
    }
    if truncated:
        result["truncated"] = True
    return result
