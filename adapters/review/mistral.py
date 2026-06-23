"""Mistral argument/red-team adapter.

Supports two providers:
  - mistral  (default) — api.mistral.ai (La Plateforme), Bearer-token auth.
  - azure             — Azure AI Studio serverless inference, Bearer-token auth.

Provider is selected via the ``provider_config`` dict passed by the pipeline.
Existing user.yaml files that specify a plain model string continue to work
unchanged.

Azure AI (serverless) configuration example (configs/user.yaml)::

    models:
      mistral:
        provider: azure
        model: mistral-large-latest     # informational; model is implicit from endpoint
        endpoint: https://Mistral-Large-abc.eastus2.inference.ai.azure.com

The Azure serverless endpoint uses the same OpenAI-compatible chat completions
format as La Plateforme — auth is ``Authorization: Bearer {key}`` in both
cases — so the only difference is the base URL.  The API key for Azure is
stored in ``api_keys.mistral.api_key`` in user.yaml as usual; just point it
to your Azure endpoint key.
"""

import json
import time
import logging
import requests

from adapters.review import streaming

DEFAULT_MODEL = "mistral-large-latest"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Inter-token read-gap timeout (seconds); constant, not the sliding-scale value.
_READ_TIMEOUT = streaming.DEFAULT_READ_TIMEOUT

# Fallback chain tried in order when primary model returns a capacity error (503).
# mistral-small-latest has reduced reasoning depth but the same API surface.
_FALLBACK_MODELS = ["mistral-small-latest"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_capacity_error(exc):
    """Return True if the exception represents a 503 capacity/availability error."""
    return "503" in str(exc)


def _resolve_model(model_arg, cfg):
    return model_arg or cfg.get("model") or DEFAULT_MODEL


def _is_reasoning_param_error(body_text):
    """Return True if a 400 body indicates an unsupported reasoning_effort parameter.

    mistral-medium-3-5-latest supports reasoning_effort; older non-magistral models reject it.
    This fallback fires when a user.yaml model override targets a standard model with a preset
    that expects reasoning support.
    """
    lower = body_text.lower()
    return "reasoning_effort" in lower and (
        "not enabled" in lower or "not supported" in lower
        or "unknown" in lower or "invalid" in lower
    )


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
    provider = cfg.get("provider", "mistral")
    # Streaming socket timeout = inter-token read-gap (small constant). The big
    # sliding-scale timeout_seconds survives only as the pipeline's wall-clock backstop.
    timeout = streaming.stream_timeout(cfg, _READ_TIMEOUT)

    if provider == "azure":
        return _call_azure(
            system_prompt, user_prompt, api_key, cfg,
            retry=retry, retry_delay=retry_delay, timeout=timeout,
        )

    # --- La Plateforme path with fallback chain ---
    requested_model = _resolve_model(model, cfg)
    reasoning_effort = cfg.get("reasoning_effort")
    models_to_try = [requested_model] + [m for m in _FALLBACK_MODELS if m != requested_model]

    result = None
    for attempt_model in models_to_try:
        result = _call_laplateforme(
            system_prompt, user_prompt, api_key,
            model=attempt_model, retry=retry, retry_delay=retry_delay,
            reasoning_effort=reasoning_effort, timeout=timeout,
        )
        if not result.get("failed"):
            if attempt_model != requested_model:
                result["fallback_from"] = requested_model
                log.warning(
                    f"Mistral used FALLBACK model {attempt_model!r} because "
                    f"{requested_model!r} was unavailable (capacity). "
                    f"Review quality may be reduced."
                )
            return result
        if _is_capacity_error(result.get("error", "")) and attempt_model != models_to_try[-1]:
            log.warning(f"Mistral {attempt_model} unavailable (capacity). Trying next fallback model.")
            continue
        return result

    return result  # exhausted fallbacks


# ---------------------------------------------------------------------------
# La Plateforme backend
# ---------------------------------------------------------------------------

def _call_laplateforme(system_prompt, user_prompt, api_key, model, retry=True, retry_delay=10, reasoning_effort=None, timeout=None):
    return _execute_request(
        system_prompt, user_prompt,
        url=MISTRAL_API_URL,
        api_key=api_key,
        model=model,
        retry=retry,
        retry_delay=retry_delay,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Azure AI (serverless inference) backend
# ---------------------------------------------------------------------------

def _call_azure(system_prompt, user_prompt, api_key, cfg, retry=True, retry_delay=10, timeout=None):
    endpoint = cfg.get("endpoint", "").rstrip("/")
    model_name = cfg.get("model", DEFAULT_MODEL)

    if not endpoint:
        return {
            "failed": True,
            "error": (
                "Azure Mistral requires 'endpoint' in the model config. "
                "See configs/user.example.yaml for the extended provider format."
            ),
            "raw": None,
            "model": model_name,
            "tokens": {},
            "elapsed_seconds": 0,
        }

    # Azure AI serverless inference appends /v1/chat/completions to the base endpoint.
    url = f"{endpoint}/v1/chat/completions"
    result = _execute_request(
        system_prompt, user_prompt,
        url=url,
        api_key=api_key,
        model=model_name,
        retry=retry,
        retry_delay=retry_delay,
        timeout=timeout,
    )
    result["provider"] = "azure"
    return result


# ---------------------------------------------------------------------------
# Shared HTTP execution (La Plateforme and Azure use the same payload format)
# ---------------------------------------------------------------------------

def _execute_request(system_prompt, user_prompt, url, api_key, model, retry, retry_delay, reasoning_effort=None, timeout=None):
    if timeout is None:
        timeout = streaming.stream_timeout(None, _READ_TIMEOUT)
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
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if with_reasoning and reasoning_effort:
            # reasoning_effort: "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
            # Supported on Magistral models; standard models reject it.
            # response_format and temperature are incompatible with reasoning mode.
            p["reasoning_effort"] = reasoning_effort
        else:
            p["response_format"] = {"type": "json_object"}
            p["temperature"] = 0.2
        return p

    session = requests.Session()
    t0 = time.monotonic()
    misconfig_msg = None

    def _post(pl):
        resp = session.post(url, headers=headers, json=pl, stream=True, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(f"Mistral {model} HTTP {resp.status_code}. Waiting {retry_delay}s before retry.")
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(url, headers=headers, json=pl, stream=True, timeout=timeout)
        resp.raise_for_status()
        return streaming.accumulate_chat_completions(resp)

    try:
        assembled = _post(_build_payload(True))
    except requests.HTTPError as e:
        elapsed = round(time.monotonic() - t0, 2)
        body = e.response.text[:400] if e.response is not None else ""
        # If the model rejected reasoning_effort, retry without it so the pass still runs.
        # This is a MISCONFIGURATION: the preset specifies a Magistral model but user.yaml
        # overrides it with a standard Mistral model. Fix by not overriding the Mistral model
        # in user.yaml for balanced+ presets, or switching to standard/economy preset.
        if reasoning_effort and _is_reasoning_param_error(body):
            misconfig_msg = (
                f"[MISCONFIGURATION] Mistral {model} rejected reasoning_effort={reasoning_effort!r}. "
                f"Reasoning is only supported on Magistral models (magistral-medium-latest, etc.). "
                f"If using a balanced+ preset, do not override the Mistral model in user.yaml "
                f"with a non-reasoning variant. Retrying without reasoning — output may be degraded."
            )
            log.error(misconfig_msg)
            try:
                assembled = _post(_build_payload(False))
            except requests.HTTPError as e2:
                elapsed = round(time.monotonic() - t0, 2)
                body2 = e2.response.text[:400] if e2.response is not None else ""
                log.error(f"Mistral {model} fallback call failed after {elapsed}s: {e2} | {body2}")
                return {
                    "failed": True, "error": str(e2), "error_body": body2,
                    "raw": None, "model": model, "tokens": {}, "elapsed_seconds": elapsed,
                }
            except Exception as e2:
                elapsed = round(time.monotonic() - t0, 2)
                log.error(f"Mistral {model} fallback call failed after {elapsed}s: {e2}")
                return {
                    "failed": True, "error": str(e2), "error_body": "",
                    "raw": None, "model": model, "tokens": {}, "elapsed_seconds": elapsed,
                }
            # fallback succeeded — assembled is set, fall through to success path below
        else:
            log.error(f"Mistral {model} call failed after {elapsed}s: {e} | {body}")
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
        log.error(f"Mistral {model} call failed after {elapsed}s: {e}")
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
    # mistral-medium-3-5 with reasoning_effort streams delta.content as a list of
    # typed chunks ([{"type":"thinking",...}, {"type":"text",...}]); the accumulator
    # keeps only the text chunks, so `content` here is already the answer text.
    content = assembled["content"]
    usage = assembled["usage"]
    if not content:
        log.warning(f"Mistral {model} returned no text content after {elapsed}s")

    def _try_parse_json(text):
        """Try to parse JSON from text, stripping markdown fences if present."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        cleaned = text.strip()
        # Strip leading markdown fences (```json or ```)
        if "```" in cleaned:
            import re
            m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", cleaned)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    pass
        return None

    parsed = _try_parse_json(content)
    if parsed is None:
        log.warning(
            f"Mistral {model} returned non-JSON content after {elapsed}s: "
            f"{content[:300]!r}"
        )
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": content,
            "model": model,
            "tokens": {
                "prompt": usage.get("prompt_tokens"),
                "completion": usage.get("completion_tokens"),
            },
            "elapsed_seconds": elapsed,
        }

    log.debug(f"Mistral {model} call succeeded in {elapsed}s ({usage.get('total_tokens', '?')} tokens)")
    result = {
        "failed": False,
        "data": parsed,
        "model": model,
        "tokens": {
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
        },
        "elapsed_seconds": elapsed,
    }
    if misconfig_msg:
        result["misconfiguration_warning"] = misconfig_msg
    return result
