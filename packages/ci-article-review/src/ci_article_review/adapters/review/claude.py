"""Anthropic Claude adapter.

Supports two thinking modes depending on model generation:

  Adaptive thinking (Opus 4.8, Opus 4.7, Fable 5, Mythos 5, Sonnet 4.6, Opus 4.6):
    Always on — the model controls its own reasoning depth.
    Configurable via the top-level ``effort`` parameter: "low", "medium", "high".
    For Sonnet 4.6 / Opus 4.6, also send ``thinking: {type: "adaptive"}`` payload key.
    Do NOT use ``thinking_budget`` on these models; extended thinking is deprecated for them.

  Extended thinking (Haiku 4.5 and earlier models):
    Opt-in via ``thinking: {type: "enabled", budget_tokens: N}``.
    Requires ``anthropic-beta: interleaved-thinking-2025-05-14`` header when thinking_budget is set.

Config examples (configs/user.yaml extended form)::

    # Opus 4.8 — adaptive thinking, control effort
    claude:
      model: claude-opus-4-8
      effort: high          # "low" | "medium" | "high"
      timeout_seconds: 360

    # Sonnet 4.6 — adaptive thinking (recommended path; extended thinking deprecated)
    claude:
      model: claude-sonnet-4-6
      effort: medium        # "low" | "medium" | "high"
      timeout_seconds: 360

    # Haiku 4.5 — extended thinking (beta header auto-added when thinking_budget is set)
    claude:
      model: claude-haiku-4-5-20251001
      thinking_budget: 5000
      timeout_seconds: 360
"""

import json
import time
import logging
import requests

from . import streaming
from ... import redact

DEFAULT_MODEL = "claude-opus-4-8"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Inter-token read-gap timeout (seconds); constant, not the sliding-scale value.
_READ_TIMEOUT = streaming.DEFAULT_READ_TIMEOUT

# Models that use adaptive thinking (always on).
# Extended thinking (thinking_budget) is deprecated or unsupported on these.
# Use the `effort` parameter to control reasoning depth.
# For Sonnet 4.6 / Opus 4.6, also send thinking: {type: "adaptive"} alongside effort.
_ADAPTIVE_THINKING_MODELS = frozenset(
    {
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-mythos-preview",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    }
)

# Models that require the interleaved-thinking beta header when thinking_budget is set.
_BETA_HEADER_MODELS = frozenset(
    {
        "claude-haiku-4-5-20251001",
    }
)

# Fallback chain tried in order when primary model returns a capacity error (529).
_FALLBACK_MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]

log = logging.getLogger(__name__)


def _redact_key(text, api_key):
    if api_key and api_key in str(text):
        return str(text).replace(api_key, "[REDACTED]")
    return str(text)


def _is_capacity_error(exc):
    msg = str(exc)
    return "529" in msg or "overloaded" in msg.lower()


def _is_adaptive_model(model):
    if model in _ADAPTIVE_THINKING_MODELS:
        return True
    for prefix in (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-fable-",
        "claude-mythos-",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ):
        if model.startswith(prefix):
            return True
    return False


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
    thinking_budget = cfg.get("thinking_budget")
    effort = cfg.get("effort")
    # Streaming socket timeout = inter-token read-gap (small constant). The big
    # sliding-scale timeout_seconds survives only as the pipeline's wall-clock backstop.
    timeout = streaming.stream_timeout(cfg, _READ_TIMEOUT)
    models_to_try = [requested_model] + [
        m for m in _FALLBACK_MODELS if m != requested_model
    ]

    for attempt_model in models_to_try:
        result = _call_model(
            system_prompt,
            user_prompt,
            api_key,
            model=attempt_model,
            retry=retry,
            retry_delay=retry_delay,
            thinking_budget=thinking_budget,
            effort=effort,
            timeout=timeout,
        )
        if not result.get("failed"):
            if attempt_model != requested_model:
                result["fallback_from"] = requested_model
                log.warning(
                    f"Claude used FALLBACK model {attempt_model!r} because "
                    f"{requested_model!r} was unavailable (capacity). "
                    f"Review quality may be reduced."
                )
            return result
        if (
            _is_capacity_error(result.get("error", ""))
            and attempt_model != models_to_try[-1]
        ):
            log.warning(
                f"Claude {attempt_model} unavailable (capacity). Trying next fallback model."
            )
            continue
        return result

    return result


def _call_model(
    system_prompt,
    user_prompt,
    api_key,
    model,
    retry=True,
    retry_delay=10,
    thinking_budget=None,
    effort=None,
    timeout=None,
):
    if timeout is None:
        timeout = streaming.stream_timeout(None, _READ_TIMEOUT)
    adaptive = _is_adaptive_model(model)

    # Warn if thinking_budget is set on an adaptive model — switch to adaptive path.
    if thinking_budget and adaptive:
        log.warning(
            f"[MISCONFIGURATION] thinking_budget is set for {model!r}, but extended thinking "
            f"is deprecated for this model. Switching to adaptive thinking path. "
            f"Use `effort` instead of `thinking_budget` in your config."
        )
        thinking_budget = None

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    # Haiku 4.5 (and other beta-header models) require the interleaved-thinking header.
    if thinking_budget and model in _BETA_HEADER_MODELS:
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        # Stream incrementally so the read timeout is the inter-token gap, not the
        # whole (potentially many-minute) generation. Usage arrives across the
        # message_start / message_delta events (see streaming.accumulate_anthropic).
        "stream": True,
    }

    if adaptive:
        # Adaptive thinking: always on, depth controlled via output_config.effort.
        # effort goes in output_config (nested), NOT as a top-level key.
        # Sending thinking: {type: "adaptive"} enables it on Sonnet 4.6 / Opus 4.6
        # (where it is off by default); harmless on Opus 4.8 / Fable 5 (always on).
        payload["max_tokens"] = 16000
        if effort:
            payload["output_config"] = {"effort": effort}
        payload["thinking"] = {"type": "adaptive"}
    elif thinking_budget:
        # Extended thinking: opt-in for Haiku 4.5 and earlier models.
        # budget_tokens must be less than max_tokens.
        payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        payload["max_tokens"] = thinking_budget + 4096
    else:
        payload["max_tokens"] = 4096

    session = requests.Session()
    t0 = time.monotonic()

    def _post():
        resp = session.post(
            CLAUDE_API_URL, headers=headers, json=payload, stream=True, timeout=timeout
        )
        if resp.status_code in (429, 500, 502, 503, 529) and retry:
            log.warning(
                f"Claude {model} HTTP {resp.status_code}. Waiting {retry_delay}s before retry."
            )
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(
                CLAUDE_API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout,
            )
        resp.raise_for_status()
        return resp

    try:
        resp = _post()
        assembled = streaming.accumulate_anthropic(resp)
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        safe_err = _redact_key(e, api_key)
        error_body = _redact_key(redact.capture_error_body(e), api_key)
        if error_body:
            log.error(
                f"Claude {model} call failed after {elapsed}s: {safe_err} | {error_body}"
            )
        else:
            log.error(f"Claude {model} call failed after {elapsed}s: {safe_err}")
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

    # accumulate_anthropic already filters thinking_delta blocks, keeping only the
    # text_delta stream — so `text` is the answer text, not reasoning.
    text = assembled["content"]
    usage = assembled["usage"]
    stop_reason = assembled["stop_reason"]

    if not text.strip():
        log.warning(
            f"Claude {model} returned no text content after {elapsed}s "
            f"(stop_reason={stop_reason!r})"
        )
        return {
            "failed": True,
            "error": f"Empty text response (stop_reason={stop_reason!r})",
            "raw": None,
            "model": model,
            "tokens": {
                "prompt": usage.get("input_tokens"),
                "completion": usage.get("output_tokens"),
            },
            "elapsed_seconds": elapsed,
        }

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning(
                f"Claude {model} returned non-JSON content after {elapsed}s: "
                f"{text[:400]!r}"
            )
            return {
                "failed": True,
                "error": "Malformed JSON response",
                "raw": text,
                "model": model,
                "tokens": usage,
                "elapsed_seconds": elapsed,
            }

    thinking_mode = (
        "adaptive" if adaptive else ("extended" if thinking_budget else "standard")
    )
    log.debug(
        f"Claude {model} call succeeded in {elapsed}s "
        f"(thinking={thinking_mode}, {usage.get('input_tokens', '?')} input tokens)"
    )
    return {
        "failed": False,
        "data": parsed,
        "model": model,
        "tokens": {
            "prompt": usage.get("input_tokens"),
            "completion": usage.get("output_tokens"),
        },
        "elapsed_seconds": elapsed,
    }
