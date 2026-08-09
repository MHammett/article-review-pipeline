import json
import time
import logging
import requests

from . import streaming
from ... import redact

DEFAULT_MODEL = "grok-4.3"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# Inter-token read-gap timeout (seconds); constant, not the sliding-scale value.
_READ_TIMEOUT = streaming.DEFAULT_READ_TIMEOUT

# Fallback chain tried in order when primary model returns a capacity error (503).
_FALLBACK_MODELS = ["grok-build-0.1"]

log = logging.getLogger(__name__)


def _redact_key(text, api_key):
    if api_key and api_key in str(text):
        return str(text).replace(api_key, "[REDACTED]")
    return str(text)


def _is_capacity_error(exc):
    """Return True if the exception represents a 503 capacity/availability error."""
    msg = str(exc)
    return "503" in msg


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
    requested_model = model or cfg.get("model") or DEFAULT_MODEL
    reasoning_effort = cfg.get("reasoning_effort")
    models_to_try = [requested_model] + [
        m for m in _FALLBACK_MODELS if m != requested_model
    ]

    # Streaming socket timeout = inter-token read-gap (small constant). The big
    # sliding-scale timeout_seconds is no longer a socket timeout; it survives only
    # as the pipeline's per-task wall-clock backstop.
    timeout = streaming.stream_timeout(cfg, _READ_TIMEOUT)

    for attempt_model in models_to_try:
        result = _call_model(
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
                    f"Grok used FALLBACK model {attempt_model!r} because "
                    f"{requested_model!r} was unavailable (capacity). "
                    f"Review quality may be reduced."
                )
            return result
        if (
            _is_capacity_error(result.get("error", ""))
            and attempt_model != models_to_try[-1]
        ):
            log.warning(
                f"Grok {attempt_model} unavailable (capacity). Trying next fallback model."
            )
            continue
        return result

    return result  # exhausted fallbacks


def _call_model(
    system_prompt,
    user_prompt,
    api_key,
    model,
    retry=True,
    retry_delay=10,
    reasoning_effort=None,
    timeout=None,
):
    if timeout is None:
        timeout = streaming.stream_timeout(None, _READ_TIMEOUT)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort:
        # reasoning_effort: "none" | "low" | "medium" | "high"
        # grok-4.3 supports this via Chat Completions; response_format can coexist with it.
        payload["reasoning_effort"] = reasoning_effort

    session = requests.Session()
    t0 = time.monotonic()

    def _post():
        resp = session.post(
            GROK_API_URL, headers=headers, json=payload, stream=True, timeout=timeout
        )
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(
                f"Grok {model} HTTP {resp.status_code}. Waiting {retry_delay}s before retry."
            )
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(
                GROK_API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout,
            )
        resp.raise_for_status()
        return resp

    try:
        resp = _post()
        assembled = streaming.accumulate_chat_completions(resp)
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        safe_err = _redact_key(e, api_key)
        error_body = _redact_key(redact.capture_error_body(e), api_key)
        if error_body:
            log.error(
                f"Grok {model} call failed after {elapsed}s: {safe_err} | {error_body}"
            )
        else:
            log.error(f"Grok {model} call failed after {elapsed}s: {safe_err}")
        session.close()
        return {
            "failed": True,
            "error": safe_err,
            "error_body": error_body,
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

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        log.warning(f"Grok {model} returned non-JSON content after {elapsed}s")
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": content,
            "model": model,
            "tokens": usage,
            "elapsed_seconds": elapsed,
        }

    log.debug(
        f"Grok {model} call succeeded in {elapsed}s ({usage.get('total_tokens', '?')} tokens)"
    )
    return {
        "failed": False,
        "data": parsed,
        "model": model,
        "tokens": {
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
        },
        "elapsed_seconds": elapsed,
    }
