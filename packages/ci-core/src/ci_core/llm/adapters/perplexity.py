"""Perplexity AI search-grounded adapter.

Perplexity's sonar models run every response through live web search before
answering — grounding is on by default, not a separately-enabled tool.  This
makes them well-suited to fact-check passes where you want a second grounded
opinion alongside Gemini.

The API is OpenAI-compatible (same endpoint shape, Bearer-token auth).

Model options:
  sonar-pro          — highest quality, best citations (default)
  sonar              — faster, slightly cheaper
  sonar-reasoning-pro — chain-of-thought reasoning + search (slower)
  sonar-reasoning    — chain-of-thought without the pro quality bump

Perplexity does not support ``response_format: json_object``.  JSON output
is requested via the system prompt; responses are occasionally wrapped in
markdown code fences, which this adapter strips before parsing.
"""

import time
import logging
import requests

from .. import schema_format, streaming
from ..tokens import normalize_tokens
from ... import redact
from ..json_utils import extract_json_with_salvage as _extract_json_with_salvage

DEFAULT_MODEL = "sonar-reasoning-pro"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Inter-token read-gap timeout (seconds); constant, not the sliding-scale value.
# sonar runs a live web search before the first token, so allow a wider gap than
# the non-grounded chat models to cover time-to-first-byte.
_READ_TIMEOUT = 160

# sonar-reasoning-pro: chain-of-thought reasoning + live search (best for fact-check)
# sonar-pro: search grounded, no reasoning chain
# sonar: fastest and cheapest, search grounded
_FALLBACK_MODELS = ["sonar-pro", "sonar"]

log = logging.getLogger(__name__)


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
    requested_model = model or cfg.get("model") or DEFAULT_MODEL
    reasoning_effort = cfg.get("reasoning_effort")
    response_schema = cfg.get("response_schema")
    search_options = cfg
    # Streaming socket timeout = inter-token read-gap (small constant). The big
    # sliding-scale timeout_seconds survives only as the pipeline's wall-clock backstop.
    timeout = streaming.stream_timeout(cfg, _READ_TIMEOUT)
    models_to_try = [requested_model] + [
        m for m in _FALLBACK_MODELS if m != requested_model
    ]

    result = None
    for attempt_model in models_to_try:
        result = _call_perplexity(
            system_prompt,
            user_prompt,
            api_key,
            model=attempt_model,
            retry=retry,
            retry_delay=retry_delay,
            reasoning_effort=reasoning_effort,
            response_schema=response_schema,
            search_options=search_options,
            timeout=timeout,
        )
        if not result.get("failed"):
            if attempt_model != requested_model:
                result["fallback_from"] = requested_model
                log.warning(
                    f"Perplexity used FALLBACK model {attempt_model!r} because "
                    f"{requested_model!r} was unavailable."
                )
            return result

        if "503" in str(result.get("error", "")) and attempt_model != models_to_try[-1]:
            log.warning(
                f"Perplexity {attempt_model} unavailable. Trying next fallback model."
            )
            continue

        return result

    return result


# ---------------------------------------------------------------------------
# HTTP execution
# ---------------------------------------------------------------------------


def _log_payload_diagnostics(model, payload):
    """Log a redacted, truncated excerpt of the outgoing payload.

    Perplexity's 400 "invalid_request" errors don't say which field is
    invalid, which made a live occurrence on the voice_style domain
    (2026-08-06) unnecessarily hard to diagnose after the fact. Called only
    on a 400/invalid_request rejection — not on every call — so this stays
    silent in the common case.
    """
    other_params = {k: v for k, v in payload.items() if k not in ("model", "messages")}
    for msg in payload.get("messages", []):
        excerpt = redact.truncate_excerpt(
            redact.redact_url_keys(msg.get("content", ""))
        )
        log.error(
            f"Perplexity {model} invalid_request payload — role={msg.get('role')} "
            f"chars={len(msg.get('content', ''))}: {excerpt}"
        )
    log.error(
        f"Perplexity {model} invalid_request payload — other params: {other_params}"
    )


def _call_perplexity(
    system_prompt,
    user_prompt,
    api_key,
    model,
    retry,
    retry_delay,
    reasoning_effort=None,
    response_schema=None,
    search_options=None,
    timeout=None,
):
    search_options = search_options or {}
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
        # Perplexity rejects response_format: json_object, which is why this
        # was prose-only for so long. It DOES accept json_schema (verified
        # live 2026-08-12) — set below when a domain schema is supplied.
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if response_schema:
        payload["response_format"] = schema_format.chat_completions_response_format(
            response_schema
        )
    # Search controls, all previously unsent. Verified live 2026-08-12 that
    # search_domain_filter genuinely constrains the citations returned (every
    # citation came back on the requested domain), so these are real levers
    # rather than hints. Opt-in per model config: a recency window aimed at the
    # "outdated" fact-check bucket, and a domain allowlist for regulatory work.
    # search_mode: "academic" restricts retrieval to scholarly sources. Measured
    # 2026-08-12 on the same query: academic returned ASME, MDPI, Taylor &
    # Francis and PLOS; the default web mode returned a LinkedIn post and a
    # university landing page. For fact-checking against peer-reviewed work
    # that is the difference between a citable source and a link.
    if search_options.get("search_mode"):
        payload["search_mode"] = search_options["search_mode"]
    for key in ("search_recency_filter", "search_after_date_filter"):
        if search_options.get(key):
            payload[key] = search_options[key]
    if search_options.get("search_domain_filter"):
        payload["search_domain_filter"] = list(search_options["search_domain_filter"])
    if search_options.get("search_context_size"):
        payload["web_search_options"] = {
            "search_context_size": search_options["search_context_size"]
        }
    if reasoning_effort:
        # Sonar API supports reasoning_effort: "minimal" | "low" | "medium" | "high"
        payload["reasoning_effort"] = reasoning_effort

    session = requests.Session()
    t0 = time.monotonic()

    def _post():
        resp = session.post(
            PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
        )
        if resp.status_code in (429, 500, 502, 503, 504) and retry:
            log.warning(
                f"Perplexity {model} HTTP {resp.status_code}. "
                f"Waiting {retry_delay}s before retry."
            )
            resp.close()
            time.sleep(retry_delay)
            resp = session.post(
                PERPLEXITY_API_URL,
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
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        elapsed = round(time.monotonic() - t0, 2)
        log.error(
            f"Perplexity {model} timed out after {elapsed}s. "
            f"For long articles, raise the perplexity timeout_seconds in user.yaml."
        )
        return {
            "failed": True,
            "error": str(e),
            "raw": None,
            "model": model,
            "tokens": {},
            "elapsed_seconds": elapsed,
            "grounding_available": False,
        }
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        error_body = redact.capture_error_body(e)
        if error_body:
            log.error(
                f"Perplexity {model} call failed after {elapsed}s: {e} | {error_body}"
            )
        else:
            log.error(f"Perplexity {model} call failed after {elapsed}s: {e}")
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 400:
            _log_payload_diagnostics(model, payload)
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
    content = assembled["content"]
    usage = assembled["usage"]
    # Citations: flat list of source URLs. search_results: richer objects with title/snippet/date.
    # Both arrive on the final SSE chunk (alongside usage) for sonar streams.
    citations = assembled["citations"]
    search_results = assembled["search_results"]
    stream_error = assembled.get("stream_error")

    # An in-band {"error": {...}} SSE event (rate limit, content rejection, etc.)
    # means no usable content was ever produced — surface that distinctly instead
    # of treating an empty/near-empty body as a JSON parse failure.
    if stream_error:
        log.warning(
            f"Perplexity {model} stream returned an error event: {stream_error}"
        )
        if isinstance(stream_error, dict) and stream_error.get("code") == 400:
            _log_payload_diagnostics(model, payload)
        return {
            "failed": True,
            "error": f"Perplexity stream error: {stream_error}",
            "raw": content or None,
            "model": model,
            "tokens": normalize_tokens(usage),
            "elapsed_seconds": elapsed,
            "grounding_available": False,
        }

    if not content:
        # No usage captured either (this is the "malformed JSON with zero tokens"
        # shape) means nothing usable ever reached the accumulator — a dropped
        # or empty stream, not a parseable-but-invalid response. Distinguishing
        # this from a genuine parse failure is the whole diagnostic point: it
        # tells you to look at the connection/stream layer, not the JSON.
        log.warning(
            f"Perplexity {model} produced no content after {elapsed}s "
            f"(usage_captured={bool(usage)})"
        )
        return {
            "failed": True,
            "error": "Empty response (no content received from Perplexity stream)",
            "raw": None,
            "model": model,
            "tokens": normalize_tokens(usage),
            "elapsed_seconds": elapsed,
            "grounding_available": False,
        }

    # Parse JSON — sonar models may prepend a <think> reasoning block or wrap
    # output in markdown code fences. _extract_json handles both plus a raw {...}
    # span; salvage then recovers the complete elements of a response that hit
    # the output-token ceiling mid-array.
    parsed, truncated = _extract_json_with_salvage(content)
    if parsed is None:
        log.warning(f"Perplexity {model} returned non-JSON content after {elapsed}s")
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": content,
            "model": model,
            "tokens": normalize_tokens(usage),
            "elapsed_seconds": elapsed,
            "grounding_available": bool(citations),
        }

    if truncated:
        log.warning(
            f"Perplexity {model} response was truncated (likely hit the output-token "
            f"ceiling — {usage.get('completion_tokens', '?')} completion tokens) after "
            f"{elapsed}s; salvaged the complete elements, discarded the rest."
        )
    else:
        log.debug(
            f"Perplexity {model} call succeeded in {elapsed}s "
            f"(citations={len(citations)}, search_results={len(search_results)}, "
            f"grounding={'yes' if citations else 'no'})"
        )
    result = {
        "failed": False,
        "raw": content,
        "data": parsed,
        "model": model,
        "tokens": normalize_tokens(usage),
        "elapsed_seconds": elapsed,
        "grounding_available": True,  # Always true for sonar models
        "citations": citations,
        "search_results": search_results,
    }
    if truncated:
        result["truncated"] = True
    return result
