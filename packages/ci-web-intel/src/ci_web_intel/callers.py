"""Multi-model caller for voice synthesis and detection.

Routes prompts to each configured provider using SSE accumulators from
adapters/review/streaming.py. Excludes Perplexity by default (web grounding
adds noise for corpus analysis tasks).

Import direction: callers.py imports from adapters/review/streaming.py ONLY.
Nothing imports back up the chain from this module.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from ci_article_review.adapters.review.streaming import (
    accumulate_anthropic,
    accumulate_chat_completions,
    accumulate_gemini,
    accumulate_openai_responses,
    stream_timeout,
)
from ci_article_review.adapters.review.json_utils import extract_json
from ci_article_review.analysis.cost import calculate as cost_calculate

log = logging.getLogger(__name__)

# Providers excluded from synthesis/detection by default
_EXCLUDED_PROVIDERS = frozenset({"perplexity"})

# Anthropic API
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_ADAPTIVE_CLAUDE_MODELS = frozenset({
    "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5",
    "claude-sonnet-4-6", "claude-opus-4-6", "claude-mythos-5",
})

# OpenAI-compatible base URLs
_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "grok": "https://api.x.ai/v1/chat/completions",
    "perplexity": "https://api.perplexity.ai/chat/completions",
}

# Gemini AI Studio
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# api_call_log accumulated across the process lifetime; read by bootstrap.py at run end
_api_call_log: list[dict] = []


def get_api_call_log() -> list[dict]:
    return list(_api_call_log)


def clear_api_call_log() -> None:
    _api_call_log.clear()


def _is_adaptive_claude(model: str) -> bool:
    return model in _ADAPTIVE_CLAUDE_MODELS or any(
        model.startswith(prefix)
        for prefix in ("claude-opus-4-8", "claude-opus-4-7", "claude-fable-",
                       "claude-sonnet-4-6", "claude-opus-4-6", "claude-mythos-")
    )


def _redact(text: Any, key: str | None) -> str:
    s = str(text)
    if key and key in s:
        s = s.replace(key, "[REDACTED]")
    return s


def _call_anthropic(
    model_name: str,
    model_cfg: dict,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    model = model_cfg.get("model", "claude-sonnet-4-6")
    effort = model_cfg.get("effort")
    thinking_budget = model_cfg.get("thinking_budget")
    timeout = stream_timeout(model_cfg)

    adaptive = _is_adaptive_claude(model)
    if thinking_budget and adaptive:
        thinking_budget = None

    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "stream": True,
    }
    if adaptive:
        payload["max_tokens"] = 16000
        if effort:
            payload["output_config"] = {"effort": effort}
        payload["thinking"] = {"type": "adaptive"}
    elif thinking_budget:
        payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        payload["max_tokens"] = thinking_budget + 4096
    else:
        payload["max_tokens"] = 8192

    t0 = time.monotonic()
    session = requests.Session()
    try:
        resp = session.post(_ANTHROPIC_URL, headers=headers, json=payload, stream=True, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 529):
            log.warning("Claude %s HTTP %d; retrying in 10s", model, resp.status_code)
            resp.close()
            time.sleep(10)
            resp = session.post(_ANTHROPIC_URL, headers=headers, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()
        assembled = accumulate_anthropic(resp)
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        err = _redact(e, api_key)
        log.error("Claude %s failed after %.1fs: %s", model, elapsed, err)
        return {"failed": True, "error": err, "model": model, "tokens": {}, "elapsed": elapsed}
    finally:
        session.close()

    elapsed = round(time.monotonic() - t0, 2)
    usage = assembled.get("usage", {})
    tokens = {
        "prompt": usage.get("input_tokens", 0),
        "completion": usage.get("output_tokens", 0),
    }
    content = assembled.get("content", "")
    log.info("%s: synthesis complete (%d tokens, %.1fs)", model_name, sum(tokens.values()), elapsed)
    return {"content": content, "failed": False, "tokens": tokens, "elapsed": elapsed, "model": model}


def _call_chat_completions(
    model_name: str,
    model_cfg: dict,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    provider: str,
) -> dict:
    model = model_cfg.get("model", "")
    timeout = stream_timeout(model_cfg)

    if provider == "azure":
        base_url = model_cfg.get("endpoint", "").rstrip("/")
        deployment = model_cfg.get("deployment", model)
        api_version = model_cfg.get("api_version", "2024-02-01")
        url = f"{base_url}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        headers = {"api-key": api_key, "Content-Type": "application/json"}
    else:
        url = _PROVIDER_URLS.get(provider, _PROVIDER_URLS["openai"])
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    reasoning_effort = model_cfg.get("reasoning_effort")
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    t0 = time.monotonic()
    session = requests.Session()
    try:
        resp = session.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503):
            log.warning("%s HTTP %d; retrying in 10s", model_name, resp.status_code)
            resp.close()
            time.sleep(10)
            resp = session.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()
        assembled = accumulate_chat_completions(resp)
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        err = _redact(e, api_key)
        log.error("%s failed after %.1fs: %s", model_name, elapsed, err)
        return {"failed": True, "error": err, "model": model, "tokens": {}, "elapsed": elapsed}
    finally:
        session.close()

    elapsed = round(time.monotonic() - t0, 2)
    usage = assembled.get("usage", {})
    tokens = {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
    }
    content = assembled.get("content", "")
    log.info("%s: synthesis complete (%d tokens, %.1fs)", model_name, sum(tokens.values()), elapsed)
    return {"content": content, "failed": False, "tokens": tokens, "elapsed": elapsed, "model": model}


def _call_gemini(
    model_name: str,
    model_cfg: dict,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    model = model_cfg.get("model", "gemini-2.5-flash")
    timeout = stream_timeout(model_cfg, default_read=160)
    thinking_budget = model_cfg.get("thinking_budget")

    url = f"{_GEMINI_BASE}/{model}:streamGenerateContent?alt=sse&key={api_key}"
    headers = {"Content-Type": "application/json"}

    gen_config: dict[str, Any] = {"temperature": 0.2, "responseMimeType": "application/json"}
    if thinking_budget is not None:
        gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget, "includeThoughts": True}

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": gen_config,
    }

    t0 = time.monotonic()
    session = requests.Session()
    try:
        resp = session.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503):
            log.warning("Gemini %s HTTP %d; retrying in 10s", model, resp.status_code)
            resp.close()
            time.sleep(10)
            resp = session.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()
        assembled = accumulate_gemini(resp)
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        err = _redact(e, api_key)
        log.error("Gemini %s failed after %.1fs: %s", model, elapsed, err)
        return {"failed": True, "error": err, "model": model, "tokens": {}, "elapsed": elapsed}
    finally:
        session.close()

    elapsed = round(time.monotonic() - t0, 2)
    usage = assembled.get("usage", {})
    tokens = {
        "prompt": usage.get("promptTokenCount", 0),
        "completion": usage.get("candidatesTokenCount", 0),
    }
    content = assembled.get("content", "")
    log.info("%s: synthesis complete (%d tokens, %.1fs)", model_name, sum(tokens.values()), elapsed)
    return {"content": content, "failed": False, "tokens": tokens, "elapsed": elapsed, "model": model}


def call_one(
    model_name: str,
    model_cfg: dict,
    api_keys: dict,
    system_prompt: str,
    user_prompt: str,
    pass_name: str = "",
) -> dict:
    """Route to the correct provider and accumulate SSE response.

    Returns {"content": str, "failed": bool, "tokens": dict, "elapsed": float, "model": str}.
    Failed providers log the error and return {"failed": True, "error": ...}.
    """
    provider = model_cfg.get("provider", model_name)
    api_key = (api_keys.get(model_name) or {}).get("api_key", "")

    if provider == "anthropic":
        result = _call_anthropic(model_name, model_cfg, api_key, system_prompt, user_prompt)
    elif provider in ("ai_studio", "vertex_ai"):
        if provider == "vertex_ai":
            log.warning("%s: Vertex AI not supported in voice profiler callers.py; using AI Studio", model_name)
        result = _call_gemini(model_name, model_cfg, api_key, system_prompt, user_prompt)
    elif provider in ("openai", "azure", "mistral", "grok", "perplexity"):
        result = _call_chat_completions(model_name, model_cfg, api_key, system_prompt, user_prompt, provider)
    else:
        log.warning("Unknown provider %r for model %s; skipping", provider, model_name)
        return {"failed": True, "error": f"Unknown provider {provider!r}", "model": model_name, "tokens": {}, "elapsed": 0.0}

    # Append to global cost log
    _api_call_log.append({
        "pass": pass_name,
        "model": result.get("model", model_cfg.get("model", model_name)),
        "tokens": result.get("tokens", {}),
        "failed": result.get("failed", False),
    })
    return result


def call_all(
    system_prompt: str,
    user_prompt: str,
    user_config: dict,
    models: list[str] | None = None,
    max_parallel: int = 0,
    exclude_perplexity: bool = True,
    pass_name: str = "",
) -> dict[str, dict]:
    """Call each model in parallel (ThreadPoolExecutor).

    Args:
        system_prompt: System prompt string.
        user_prompt: User prompt string.
        user_config: Full user config (from load_user_config / _load_yaml + normalize).
        models: Explicit model subset to use; None = all configured models.
        max_parallel: Max concurrent threads; 0 = all at once.
        exclude_perplexity: If True, skip perplexity (default True).
        pass_name: Label for cost log.

    Returns:
        {model_name: {"content": str, "failed": bool, "tokens": dict, "elapsed": float}}
    """
    models_cfg: dict = user_config.get("models", {})
    api_keys: dict = user_config.get("api_keys", {})

    # Normalize model configs (handle simple string form)
    from ci_article_review.config_loader import _normalize_model_configs
    models_cfg = _normalize_model_configs(models_cfg)

    # Filter to requested subset
    if models is not None:
        active = {k: v for k, v in models_cfg.items() if k in models}
    else:
        active = dict(models_cfg)

    # Exclude disabled models
    active = {k: v for k, v in active.items() if v.get("enabled", True) is not False}

    # Exclude perplexity by default
    if exclude_perplexity:
        active = {k: v for k, v in active.items() if k != "perplexity"}

    if not active:
        log.warning("call_all: no active models to call (models=%r, exclude_perplexity=%r)", models, exclude_perplexity)
        return {}

    workers = len(active) if max_parallel <= 0 else max_parallel

    results: dict[str, dict] = {}

    def _task(name: str, cfg: dict) -> tuple[str, dict]:
        return name, call_one(name, cfg, api_keys, system_prompt, user_prompt, pass_name=pass_name)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, name, cfg): name for name, cfg in active.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                _, result = fut.result()
                results[name] = result
            except Exception as e:
                log.error("call_all: unexpected error from %s: %s", name, e)
                results[name] = {"failed": True, "error": str(e), "tokens": {}, "elapsed": 0.0}

    return results


def log_cost_summary() -> None:
    """Log spend summary using the accumulated api_call_log."""
    log_entries = get_api_call_log()
    if not log_entries:
        return
    summary = cost_calculate(log_entries)
    log.info(
        "API spend: $%.4f total ($%.4f input + $%.4f output) across %d calls%s",
        summary["total_usd"],
        summary["total_input_usd"],
        summary["total_output_usd"],
        len(log_entries),
        "" if summary["pricing_known"] else " [pricing estimate — some models unknown]",
    )
