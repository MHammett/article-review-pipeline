"""OpenAI voice/completeness adapter.

Supports three providers:
  - openai       (default) — api.openai.com, Bearer-token auth.
  - openai_search          — api.openai.com Responses API with web_search_preview tool.
                             Set ``web_search: true`` in the model config to enable.
                             Falls back to standard chat completions on error.
  - azure                  — Azure OpenAI Service, api-key header auth.

Provider is selected via the ``provider_config`` dict passed by the pipeline.
Existing user.yaml files that specify a plain model string continue to work
unchanged.

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

import json
import time
import logging
import requests

from . import streaming

DEFAULT_MODEL = "gpt-5.4"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

# Inter-token read-gap timeout (seconds) for the streaming socket. Constant, not
# derived from the sliding-scale timeout_seconds — see adapters/review/streaming.py.
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
            system_prompt, user_prompt, api_key, cfg,
            retry=retry, retry_delay=retry_delay, timeout=timeout,
        )

    # web_search: true → try Responses API with live search, fall back if unavailable
    if cfg.get("web_search"):
        requested_model = _resolve_model(model, cfg)
        result = _call_with_web_search(
            system_prompt, user_prompt, api_key,
            model=requested_model, retry=retry, retry_delay=retry_delay,
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
    models_to_try = [requested_model] + [m for m in _FALLBACK_MODELS if m != requested_model]

    result = None
    for attempt_model in models_to_try:
        result = _call_openai(
            system_prompt, user_prompt, api_key,
            model=attempt_model, retry=retry, retry_delay=retry_delay,
            reasoning_effort=reasoning_effort, timeout=timeout,
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
        if _is_capacity_error(result.get("error", "")) and attempt_model != models_to_try[-1]:
            log.warning(f"OpenAI {attempt_model} unavailable (capacity). Trying next fallback model.")
            continue
        return result

    return result  # exhausted fallbacks


# ---------------------------------------------------------------------------
# OpenAI Responses API (web search)
# ---------------------------------------------------------------------------

def _call_with_web_search(system_prompt, user_prompt, api_key, model, retry, retry_delay, timeout=None):
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
        resp = session.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, stream=True, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(
                f"OpenAI Responses API {model} HTTP {resp.status_code}. "
                f"Waiting {retry_delay}s before retry."
            )
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()
        return resp

    try:
        resp = _post()
        assembled = streaming.accumulate_openai_responses(resp)
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        log.warning(f"OpenAI Responses API {model} call failed after {elapsed}s: {e}")
        session.close()
        return {
            "failed": True,
            "error": str(e),
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
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning(f"OpenAI (web search) {model} returned non-JSON content after {elapsed}s")
            return {
                "failed": True,
                "error": "Malformed JSON response",
                "raw": content,
                "model": model,
                "tokens": usage,
                "elapsed_seconds": elapsed,
                "grounding_available": True,
            }

    log.debug(f"OpenAI (web search) {model} call succeeded in {elapsed}s")
    return {
        "failed": False,
        "data": parsed,
        "model": f"{model}+search",
        "tokens": {
            "prompt": usage.get("input_tokens"),
            "completion": usage.get("output_tokens"),
        },
        "elapsed_seconds": elapsed,
        "grounding_available": True,
    }


# ---------------------------------------------------------------------------
# openai.com backend
# ---------------------------------------------------------------------------

def _call_openai(system_prompt, user_prompt, api_key, model, retry=True, retry_delay=10, reasoning_effort=None, timeout=None):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _build_payload(with_reasoning):
        p = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Stream tokens incrementally; ask for usage in the final chunk so cost
            # estimation still works (usage is omitted from streamed responses
            # unless include_usage is requested).
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if with_reasoning and reasoning_effort:
            # reasoning_effort: "none" | "low" | "medium" | "high" | "max"
            # Supported on o-series and reasoning-capable models.
            # response_format and temperature are incompatible with reasoning mode.
            p["reasoning_effort"] = reasoning_effort
        else:
            p["response_format"] = {"type": "json_object"}
            p["temperature"] = 0.2
        return p

    result = _execute_request(
        headers, _build_payload(True), model, api_key=api_key,
        url=OPENAI_API_URL, retry=retry, retry_delay=retry_delay,
        timeout=timeout,
    )

    # If the model rejected reasoning_effort, retry without it so the pass still runs.
    # This is a MISCONFIGURATION: the preset specifies a reasoning model (o4-mini, o3)
    # but user.yaml overrides it with a non-reasoning model. Fix by not overriding the
    # model in user.yaml for balanced+ presets, or switching to a standard/economy preset.
    if (reasoning_effort and result.get("failed")
            and _is_reasoning_param_error(result.get("error_body", ""))):
        msg = (
            f"[MISCONFIGURATION] OpenAI {model} rejected reasoning_effort={reasoning_effort!r}. "
            f"Reasoning is only supported on o-series models (o4-mini, o3, etc.). "
            f"If using a balanced+ preset, do not override the OpenAI model in user.yaml "
            f"with a non-reasoning variant. Retrying without reasoning — output may be degraded."
        )
        log.error(msg)
        retry_result = _execute_request(
            headers, _build_payload(False), model, api_key=api_key,
            url=OPENAI_API_URL, retry=retry, retry_delay=retry_delay,
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

def _call_azure(system_prompt, user_prompt, api_key, cfg, retry=True, retry_delay=10, timeout=None):
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
        "model": deployment,   # Azure accepts this; ignored by the service but harmless
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
        headers, payload, model_name, api_key=None,
        url=url, retry=retry, retry_delay=retry_delay,
        timeout=timeout,
    )
    # Tag so callers know which endpoint was used
    result["provider"] = "azure"
    return result


# ---------------------------------------------------------------------------
# Shared HTTP execution
# ---------------------------------------------------------------------------

def _execute_request(headers, payload, model, api_key, url, retry, retry_delay, timeout=None):
    if timeout is None:
        timeout = streaming.stream_timeout(None, _READ_TIMEOUT)
    session = requests.Session()
    t0 = time.monotonic()

    def _post():
        resp = session.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(f"OpenAI {model} HTTP {resp.status_code}. Waiting {retry_delay}s before retry.")
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()
        return resp

    try:
        resp = _post()
        assembled = streaming.accumulate_chat_completions(resp)
    except requests.HTTPError as e:
        elapsed = round(time.monotonic() - t0, 2)
        body = e.response.text[:400] if e.response is not None else ""
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

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        log.warning(f"OpenAI {model} returned non-JSON content after {elapsed}s")
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": content,
            "model": model,
            "tokens": usage,
            "elapsed_seconds": elapsed,
        }

    log.debug(f"OpenAI {model} call succeeded in {elapsed}s ({usage.get('total_tokens', '?')} tokens)")
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
