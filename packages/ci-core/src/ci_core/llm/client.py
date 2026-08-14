"""The litellm shim — one call path to every provider.

This replaces the six hand-rolled adapters, the raw-SSE accumulators, and the
token-normalisation table that preceded it (~3,350 lines). What is left here is
the part litellm does not do: mapping our call contract onto it, keeping our own
retry policy, and pulling the provider-specific fields the pipeline reads back
out of the response.

Two call surfaces, not one
--------------------------
Five providers go through :func:`litellm.completion`. OpenAI goes through
:func:`litellm.responses`, and that is not a style preference — it is measured.
``completion()`` routes reasoning models through Chat Completions, which sends
**zero bytes** while the model thinks: same model, effort, and prompt, TTFB was
79.1s with 100% of the call silent. ``responses()`` streams reasoning summaries
instead — TTFB 0.8s, 1318 summary deltas. Since the read-gap timeout below is
the only thing standing between a slow call and a hung one, a surface that goes
silent for 79s is unusable. Do not "simplify" OpenAI onto ``completion()``.

The read-gap timeout
--------------------
Every call streams, so the HTTP read timeout is the gap *between* chunks, not
the total generation budget. A 20-minute answer that emits a token every second
is fine; 130 seconds of silence is not. That distinction is what lets the
timeouts stay small and constant while models get slower.

:func:`_stream_timeout` builds an ``httpx.Timeout``. litellm passes it through
intact for OpenAI and Azure; for the other five providers it coerces the object
down to ``float(timeout.read)`` (``CompletionTimeout.resolve``). That coercion
is harmless and was verified in the spike: httpx applies a bare float
per-operation, so ``read`` still means "gap between reads" and gemini/mistral
completed 7-second calls under a 3-second read timeout without tripping. The
only thing lost is the separate 30s connect bound — those five inherit the read
value as their connect timeout, which is more permissive, never less.

Retries stay ours
-----------------
``num_retries=0`` on every call. litellm classifies credit exhaustion as
``RateLimitError`` — measured: an OpenAI account with no credits comes back as a
429, identical by status to a per-minute limit a short wait would clear. Its own
retry logic would therefore sit and retry a dead account until the wall-clock
backstop fires. Our policy — one retry on a genuinely transient failure, then
give up and report — is applied in :func:`_with_retry`, with the narrow
terminal-vs-transient check the 429 makes unavoidable. The full classifier
belongs upstream, where each provider's exhaustion wording is better known than
it is here.

Two known gaps in litellm that this file works around, both worth reporting
upstream: the credit-exhaustion classification above, and litellm's per-provider
parameter allowlist rejecting ``reasoning_effort`` for Mistral even though the
model accepts it (see :func:`_provider_params`).

Result contract
---------------
Unchanged from the adapters, because the pipeline, ci-style-profile, and the
report generator all read it:

  ``failed``           bool
  ``raw``              assembled response text
  ``data``             parsed JSON payload (success only)
  ``model``            the model that actually answered
  ``tokens``           ``{"prompt": int, "completion": int, "cached": int}``
  ``elapsed_seconds``  float
  ``error``            redacted exception text (failures only)
  ``error_body``       redacted excerpt of the HTTP error body

plus per-provider extras: ``citations`` / ``search_results`` (Perplexity),
``grounding_chunks`` / ``grounding_available`` (Gemini), ``truncated``,
``fallback_from``, ``misconfiguration_warning``.
"""

import logging
import time

import httpx
import litellm

from .. import redact
from .json_utils import extract_json_with_salvage

log = logging.getLogger(__name__)

# litellm prints a "provider list" banner and update nags on first use, and logs
# a two-line "LiteLLM completion() model=..." banner at INFO for every call — 30
# calls a run, doubled, interleaved with the pipeline's own progress output. This
# is a library inside a CLI whose stdout is a report an operator reads; its
# warnings and errors still come through, and -v still raises the level back.
litellm.suppress_debug_info = True
litellm.telemetry = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Connection-establishment timeout. Small and constant — opening a socket has
# nothing to do with how long the model will think. Honored as-is for OpenAI
# and Azure; see the module docstring for what the other five do with it.
DEFAULT_CONNECT_TIMEOUT = 30

# Default inter-chunk read gap when a model config sets no ``stream_read_timeout``.
# NOT the total generation budget. Grounded models search before emitting their
# first token, so they default higher.
DEFAULT_READ_TIMEOUT = 120
GROUNDED_READ_TIMEOUT = 160

# HTTP statuses worth one retry. Everything else is reported as-is: retrying a
# 400 or a 401 just spends the budget twice to learn the same thing.
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# Providers that accept a temperature. Anthropic's reasoning models reject any
# value other than 1 outright — `claude-opus-4-8 does not support
# temperature=0.2` is a hard 400, so sending it broke every Claude call in the
# ensemble. The adapter this replaced never sent one to Anthropic either; this
# preserves that. litellm.drop_params would also silence it, but globally, which
# would hide the next parameter mismatch instead of surfacing it.
_SENDS_TEMPERATURE = frozenset({"gemini", "mistral", "grok", "perplexity"})
_TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# Provider table
# ---------------------------------------------------------------------------
#
# ``prefix`` is litellm's provider routing prefix. ``surface`` picks the call
# shape (see the module docstring on why OpenAI differs). ``fallbacks`` is tried
# in order when the primary model returns a capacity error, and a fallback that
# answers is reported loudly — a quietly degraded review is worse than a failed
# one, because it still produces findings that look authoritative.

_PROVIDERS = {
    "openai": {
        "prefix": "",
        "surface": "responses",
        "default_model": "gpt-5.4",
        "fallbacks": ["gpt-5.4-mini"],
        "read_timeout": DEFAULT_READ_TIMEOUT,
    },
    "gemini": {
        "prefix": "gemini/",
        "surface": "completion",
        "default_model": "gemini-2.5-flash",
        "fallbacks": ["gemini-2.5-flash-lite"],
        "read_timeout": GROUNDED_READ_TIMEOUT,
    },
    "mistral": {
        "prefix": "mistral/",
        "surface": "completion",
        "default_model": "mistral-large-latest",
        "fallbacks": ["mistral-small-latest"],
        "read_timeout": DEFAULT_READ_TIMEOUT,
    },
    "grok": {
        "prefix": "xai/",
        "surface": "completion",
        "default_model": "grok-4.3",
        "fallbacks": ["grok-build-0.1"],
        "read_timeout": DEFAULT_READ_TIMEOUT,
    },
    "claude": {
        "prefix": "anthropic/",
        "surface": "completion",
        "default_model": "claude-opus-4-8",
        "fallbacks": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "read_timeout": DEFAULT_READ_TIMEOUT,
    },
    "perplexity": {
        "prefix": "perplexity/",
        "surface": "completion",
        "default_model": "sonar-reasoning-pro",
        "fallbacks": ["sonar-pro", "sonar"],
        "read_timeout": GROUNDED_READ_TIMEOUT,
    },
}

PROVIDERS = tuple(_PROVIDERS)


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


def _stream_timeout(cfg, default_read):
    """Build the streaming timeout. ``read`` is the inter-chunk gap.

    A per-model ``stream_read_timeout`` overrides the provider default. The
    sliding-scale ``timeout_seconds`` is deliberately NOT used here — that value
    is the pipeline's per-task wall-clock backstop, enforced by a thread
    wrapper, and using it as a socket timeout would defeat the whole point of
    streaming (a long-but-healthy generation would be killed).
    """
    read = float((cfg or {}).get("stream_read_timeout") or default_read)
    return httpx.Timeout(
        connect=float(DEFAULT_CONNECT_TIMEOUT), read=read, write=read, pool=read
    )


def _resolve_model(provider, model_arg, cfg):
    return (
        model_arg or (cfg or {}).get("model") or _PROVIDERS[provider]["default_model"]
    )


def _qualified(provider, model):
    """Prefix a bare model id for litellm's provider routing.

    A model that already carries a ``provider/`` prefix is left alone so an
    operator can pin an exact litellm route in user.yaml.
    """
    prefix = _PROVIDERS[provider]["prefix"]
    if not prefix or "/" in model:
        return model
    return f"{prefix}{model}"


def _provider_params(provider, cfg):
    """Per-provider request parameters drawn from the model config.

    Only parameters the presets actually set are mapped. Anything unrecognised
    in the model config is left out rather than forwarded blindly — a stray key
    reaching a provider is a 400 mid-run, which costs a whole domain's review.
    """
    cfg = cfg or {}
    params = {}

    if provider == "gemini":
        # Search grounding. This is the entire reason gemini is in the fact_check
        # ensemble; without it the model is answering from training recall.
        params["tools"] = [{"googleSearch": {}}]
        budget = cfg.get("thinking_budget")
        if budget is not None:
            params["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}

    elif provider == "claude":
        budget = cfg.get("thinking_budget")
        effort = cfg.get("effort")
        if budget is not None:
            params["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}
            params["max_tokens"] = int(budget) + 4096
        elif effort:
            params["reasoning_effort"] = effort
            params["max_tokens"] = 16000
        else:
            params["max_tokens"] = 4096
        # No temperature — see _SENDS_TEMPERATURE.

    elif provider in ("mistral", "grok", "perplexity"):
        effort = cfg.get("reasoning_effort")
        if effort:
            params["reasoning_effort"] = effort
            if provider == "mistral":
                # litellm's per-provider allowlist does not include
                # reasoning_effort for Mistral, so it raises UnsupportedParamsError
                # before the request is ever sent — client-side, in 0.05s. The
                # parameter is real: mistral-medium-3-5 accepts "high" and "none"
                # and 400s on "low"/"medium", which is a distinction only the
                # provider could be making. Verified 2026-08-14 that the call
                # succeeds with reasoning once the parameter is allowed through.
                #
                # This is the narrow fix rather than litellm.drop_params=True.
                # drop_params would make the call succeed by silently discarding
                # reasoning, turning a loud failure into a quiet quality
                # regression across all five domains — and it is global, so it
                # would hide the next parameter mismatch too.
                params["allowed_openai_params"] = ["reasoning_effort"]
        if provider == "mistral":
            params["max_tokens"] = int(cfg.get("max_tokens", 8000))

    # Perplexity search controls, when an operator sets them.
    for key in ("search_mode", "search_recency_filter", "search_domain_filter"):
        if provider == "perplexity" and cfg.get(key) is not None:
            params[key] = cfg[key]

    return params


# ---------------------------------------------------------------------------
# Response reading
# ---------------------------------------------------------------------------


def _text_of(delta):
    """Pull text out of a streamed delta.

    Mistral's reasoning models send ``content`` as a list of typed chunks
    (``{"type": "thinking"}`` / ``{"type": "text"}``) rather than a string; only
    the text chunks belong in the answer.
    """
    piece = getattr(delta, "content", None)
    if isinstance(piece, list):
        out = []
        for part in piece:
            get = (
                part.get
                if isinstance(part, dict)
                else lambda k, d=None: getattr(part, k, d)
            )
            if get("type") == "text":
                out.append(get("text", "") or "")
        return "".join(out)
    return piece or ""


def _consume_completion_stream(stream):
    """Drain a ``litellm.completion(stream=True)`` response.

    Returns the assembled text plus everything the pipeline reads off the
    response: usage, finish reason, and provider extras.

    Note the grounding handling. Gemini emits ``vertex_ai_grounding_metadata``
    on more than one chunk, and the early ones are empty ``{}`` — the populated
    object with ``groundingChunks`` arrives last. Taking the first non-None
    value therefore yields nothing at all, silently, on a call that really did
    ground. Last-non-empty wins.
    """
    parts = []
    usage = None
    finish_reason = None
    citations = []
    search_results = []
    grounding_meta = {}

    for chunk in stream:
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            if delta is not None:
                parts.append(_text_of(delta))
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage

        cites = getattr(chunk, "citations", None)
        if cites:
            citations = list(cites)
        results = getattr(chunk, "search_results", None)
        if results:
            search_results = list(results)

        meta = getattr(chunk, "vertex_ai_grounding_metadata", None)
        if meta:
            for entry in meta if isinstance(meta, list) else [meta]:
                if isinstance(entry, dict) and entry:
                    grounding_meta = entry

    return {
        "content": "".join(parts),
        "usage": usage,
        "finish_reason": finish_reason,
        "citations": citations,
        "search_results": search_results,
        "grounding_metadata": grounding_meta,
    }


def _consume_responses_stream(stream):
    """Drain a ``litellm.responses(stream=True)`` response (OpenAI).

    The Responses API streams typed events. ``response.output_text.delta``
    carries answer text; ``response.completed`` carries final usage. The
    ``response.reasoning_summary_text.delta`` events carry no answer content,
    but reading them off the socket is what resets the read-gap timer during
    the model's thinking phase — which is the whole reason this surface exists.
    Everything else is ignored; not crashing on it is enough.
    """
    parts = []
    reasoning_parts = []
    usage = None
    status = None

    for event in stream:
        etype = getattr(event, "type", None)
        if etype == "response.output_text.delta":
            parts.append(getattr(event, "delta", "") or "")
        elif etype == "response.reasoning_summary_text.delta":
            reasoning_parts.append(getattr(event, "delta", "") or "")
        elif etype in ("response.completed", "response.incomplete"):
            resp = getattr(event, "response", None)
            if resp is not None:
                usage = getattr(resp, "usage", None) or usage
                status = getattr(resp, "status", None) or status
                incomplete = getattr(resp, "incomplete_details", None)
                if incomplete is not None:
                    status = getattr(incomplete, "reason", None) or status

    return {
        "content": "".join(parts),
        "usage": usage,
        # The Responses API reports truncation as an incomplete status with
        # reason "max_output_tokens" rather than finish_reason="length".
        "finish_reason": "length" if status == "max_output_tokens" else status,
        "reasoning_summary": "".join(reasoning_parts),
        "citations": [],
        "search_results": [],
        "grounding_metadata": {},
    }


def _read_tokens(usage):
    """Normalise litellm's usage object to ``{prompt, completion, cached}``.

    litellm already reconciles the providers' spelling disagreements
    (``promptTokenCount`` / ``input_tokens`` / ``prompt_tokens``) and — verified
    against a live Gemini call — already folds thinking tokens into
    ``completion_tokens``, which the old table had to add by hand. So this is
    only about reading the cached count off the details object and tolerating a
    provider that sends no usage at all.
    """
    if usage is None:
        return {"prompt": 0, "completion": 0, "cached": 0}

    def _int(value):
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt = _int(
        getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    )
    completion = _int(
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
    )

    details = getattr(usage, "prompt_tokens_details", None)
    cached = _int(getattr(details, "cached_tokens", None)) if details else 0
    if not cached:
        # Anthropic reports cache hits on its own key rather than in the details
        # object; litellm passes it through under the same name.
        cached = _int(getattr(usage, "cache_read_input_tokens", None))

    return {"prompt": prompt, "completion": completion, "cached": cached}


def _grounding_chunks(metadata):
    """Extract ``[{"uri", "title"}, ...]`` from Gemini's grounding metadata.

    This shape is a contract, not an internal detail: ``resolve_grounding_urls``
    consumes it, and every ``uri`` here is a
    ``vertexaisearch.cloud.google.com/grounding-api-redirect/...`` wrapper that
    expires in roughly 30 days. They must be resolved to real source URLs before
    anything stores one — a citation pointing at a dead redirect looks like a
    source and is not one.
    """
    out = []
    for chunk in (metadata or {}).get("groundingChunks") or []:
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if isinstance(web, dict) and web.get("uri"):
            out.append({"uri": web["uri"], "title": web.get("title", "")})
    return out


# ---------------------------------------------------------------------------
# Errors and retry
# ---------------------------------------------------------------------------


def _status_of(exc):
    """HTTP status behind a litellm exception, or None."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


# A dead account is not a rate limit, whatever the status code says.
#
# Measured against litellm 1.96.2 on 2026-08-14: an OpenAI account with no
# credits raises RateLimitError with status_code 429 — indistinguishable by
# status from a genuine per-minute limit that a short wait would clear. Retrying
# it is pure waste: it fails again, having spent retry_delay to learn nothing,
# and on a 30-call run that is 30 pointless waits.
#
# This is the narrow, local half of the terminal-vs-transient classifier. The
# full version is being offered upstream (UPSTREAM.md #1), where it belongs —
# litellm is better placed to know each provider's exhaustion wording than we
# are. What is kept here is only enough to stop our own single retry.
_TERMINAL_QUOTA_MARKERS = (
    "no credits remaining",
    "insufficient credit",
    "insufficient_quota",
    "exceeded your current quota",
    "billing",
    "payment required",
    "account is not active",
)


def _is_terminal_quota_error(exc):
    """True when a 429 means the wallet is empty, not that we went too fast.

    Reads the response body as well as the exception text: litellm folds the
    provider's message into the exception for OpenAI, but others put it only in
    the body, and this has to be right for all of them.
    """
    parts = [str(exc), str(getattr(exc, "message", "") or "")]
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            parts.append(str(response.text or ""))
        except Exception:
            pass
    text = " ".join(parts).lower()
    return any(marker in text for marker in _TERMINAL_QUOTA_MARKERS)


def _is_capacity_error(status):
    """True when the provider is out of capacity for this model — worth a fallback.

    Takes the recorded status rather than the exception text. Substring-matching
    "503" in a message is wrong now that litellm wraps mid-stream failures in
    ``MidStreamFallbackError``, which synthesises status 503 when the underlying
    error carries none: a dropped socket would read as "model at capacity" and
    silently walk the fallback chain, downgrading the model over a network blip.
    """
    return status == 503


def _error_body(exc):
    """Redacted excerpt of a provider's error body.

    The status line alone cannot distinguish an invalid key from a revoked one
    or from an account out of credit; providers put that in the body. Gemini
    puts the API key in the URL query string, so this goes through
    ``redact_url_keys`` before it can reach a log or a report.
    """
    body = ""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = response.text or ""
        except Exception:
            body = ""
    if not body:
        body = getattr(exc, "message", "") or ""
    return redact.truncate_excerpt(redact.redact_url_keys(body)) if body else ""


def _with_retry(fn, retry, retry_delay, label):
    """Run ``fn``, retrying once on a genuinely transient failure.

    ``num_retries=0`` is set on the litellm call itself, so this is the only
    retry in the stack. That is deliberate: litellm reports credit exhaustion as
    a ``RateLimitError``, and its retry loop would treat a dead account as a
    transient limit and keep going until the wall-clock backstop killed the run.

    The case this exists for, measured: a dropped keepalive connection surfaces
    as ``InternalServerError`` (status 500) carrying ``[WinError 10038] An
    operation was attempted on something that is not a socket`` — the cached
    HTTP client handing back a connection the provider had already closed. It
    showed up on roughly 10% of same-host calls made a second or two apart,
    which is exactly the spacing this pipeline uses. A new socket fixes it
    completely, so it must be retried; without that, a recoverable blip costs a
    whole domain's review.
    """
    try:
        return fn()
    except Exception as exc:
        status = _status_of(exc)
        if not retry or status not in _RETRYABLE_STATUS:
            raise
        # Checked against every retryable status, not just 429. An exhausted
        # account reaches us as a 429 directly, but mid-stream it arrives
        # wrapped in MidStreamFallbackError, which synthesises 503 — same dead
        # wallet, different number. The test is what the provider said, not
        # which status code the wrapper picked.
        if _is_terminal_quota_error(exc):
            log.error(
                f"{label} reported HTTP {status}, but the account is exhausted. "
                f"Not retrying — a wait will not refill it."
            )
            raise
        log.warning(f"{label} HTTP {status}. Waiting {retry_delay}s before one retry.")
        time.sleep(retry_delay)
        return fn()


def _is_reasoning_param_error(body_text):
    """True if a 400 body says the reasoning parameter is unsupported."""
    lower = (body_text or "").lower()
    return (
        "unknown_parameter" in lower or "unsupported parameter" in lower
    ) and "reasoning" in lower


# ---------------------------------------------------------------------------
# One attempt
# ---------------------------------------------------------------------------


def _attempt(
    provider,
    model,
    system_prompt,
    user_prompt,
    api_key,
    cfg,
    timeout,
    retry,
    retry_delay,
    with_reasoning=True,
):
    """One model call, start to finish, as a result dict. Never raises."""
    spec = _PROVIDERS[provider]
    params = _provider_params(provider, cfg) if with_reasoning else {}
    label = f"{provider} {model}"
    t0 = time.monotonic()

    def _invoke():
        if spec["surface"] == "responses":
            kwargs = {
                "model": _qualified(provider, model),
                "instructions": system_prompt,
                "input": user_prompt,
                "stream": True,
                "timeout": timeout,
                "num_retries": 0,
                "api_key": api_key,
            }
            effort = (cfg or {}).get("reasoning_effort")
            if with_reasoning and effort:
                # summary="auto" is what makes the thinking phase audible on the
                # wire; without it this surface goes as quiet as Chat Completions.
                kwargs["reasoning"] = {"effort": effort, "summary": "auto"}
            else:
                kwargs["temperature"] = 0.2
            return _consume_responses_stream(litellm.responses(**kwargs))

        if provider in _SENDS_TEMPERATURE:
            params.setdefault("temperature", _TEMPERATURE)

        return _consume_completion_stream(
            litellm.completion(
                model=_qualified(provider, model),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                stream_options={"include_usage": True},
                timeout=timeout,
                num_retries=0,
                api_key=api_key,
                **params,
            )
        )

    try:
        assembled = _with_retry(_invoke, retry, retry_delay, label)
    except Exception as exc:
        elapsed = round(time.monotonic() - t0, 2)
        body = _error_body(exc)
        safe = redact.redact_url_keys(exc)
        log.error(
            f"{label} call failed after {elapsed}s: {safe}"
            + (f" | {body}" if body else "")
        )
        return {
            "failed": True,
            "error": safe,
            "error_body": body,
            "raw": None,
            "model": model,
            "tokens": {"prompt": 0, "completion": 0, "cached": 0},
            "elapsed_seconds": elapsed,
            "grounding_available": False,
            # Internal: the fallback decision reads these rather than
            # substring-matching the message. Stripped before the result
            # reaches the report.
            "_status": _status_of(exc),
            "_terminal": _is_terminal_quota_error(exc),
        }

    elapsed = round(time.monotonic() - t0, 2)
    content = assembled["content"]
    tokens = _read_tokens(assembled["usage"])
    grounding = _grounding_chunks(assembled["grounding_metadata"])
    citations = assembled["citations"]

    extras = {}
    if provider == "perplexity":
        extras["citations"] = citations
        extras["search_results"] = assembled["search_results"]
        extras["grounding_available"] = bool(citations)
    elif provider == "gemini":
        extras["grounding_chunks"] = grounding
        extras["grounding_available"] = bool(grounding)

    parsed, truncated = extract_json_with_salvage(content)
    if parsed is None:
        hit_ceiling = assembled["finish_reason"] == "length"
        log.warning(
            f"{label} returned non-JSON content after {elapsed}s"
            + (" (cut off at the output-token ceiling)" if hit_ceiling else "")
        )
        return {
            "failed": True,
            "error": "Malformed JSON response",
            "raw": content,
            "model": model,
            "tokens": tokens,
            "elapsed_seconds": elapsed,
            **extras,
        }

    # finish_reason="length" means the ceiling cut the response off even when
    # the salvage path recovered parseable JSON from what arrived.
    truncated = truncated or assembled["finish_reason"] == "length"
    if truncated:
        log.warning(
            f"{label} response was truncated (hit the output-token ceiling at "
            f"{tokens['completion']} output tokens) after {elapsed}s; kept the "
            f"complete elements, discarded the rest."
        )
    else:
        log.debug(f"{label} call succeeded in {elapsed}s")

    result = {
        "failed": False,
        "raw": content,
        "data": parsed,
        "model": model,
        "tokens": tokens,
        "elapsed_seconds": elapsed,
        **extras,
    }
    if truncated:
        result["truncated"] = True
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _strip_internal(result):
    """Drop the routing-only keys before the result reaches a caller.

    The pipeline writes this dict into the run report; an underscore-prefixed
    key would show up there as if it meant something to a reader.
    """
    for key in ("_status", "_terminal"):
        result.pop(key, None)
    return result


def call(
    provider,
    system_prompt,
    user_prompt,
    api_key,
    retry=True,
    retry_delay=10,
    model=None,
    provider_config=None,
):
    """Call ``provider`` and return the shared result dict.

    Walks the fallback chain on capacity errors and retries once without
    reasoning parameters when a model rejects them.
    """
    if provider not in _PROVIDERS:
        raise KeyError(
            f"Unknown provider {provider!r}. Known providers: "
            f"{', '.join(sorted(_PROVIDERS))}"
        )

    cfg = provider_config or {}
    requested = _resolve_model(provider, model, cfg)
    timeout = _stream_timeout(cfg, _PROVIDERS[provider]["read_timeout"])
    chain = [requested] + [
        m for m in _PROVIDERS[provider]["fallbacks"] if m != requested
    ]

    result = None
    for attempt_model in chain:
        result = _attempt(
            provider,
            attempt_model,
            system_prompt,
            user_prompt,
            api_key,
            cfg,
            timeout,
            retry,
            retry_delay,
        )

        # A model that rejects the reasoning parameter is a misconfiguration, not
        # a provider fault: the preset asked for a reasoning model and user.yaml
        # overrode it with one that cannot reason. Retry without it so the domain
        # still gets reviewed, but say so loudly — the output is degraded and the
        # operator needs to fix the config, not the run.
        if (
            result.get("failed")
            and cfg.get("reasoning_effort")
            and _is_reasoning_param_error(result.get("error_body", ""))
        ):
            msg = (
                f"[MISCONFIGURATION] {provider} {attempt_model} rejected "
                f"reasoning_effort={cfg.get('reasoning_effort')!r}. Do not override the "
                f"{provider} model in user.yaml with a non-reasoning variant on a "
                f"balanced+ preset. Retrying without reasoning — output may be degraded."
            )
            log.error(msg)
            result = _attempt(
                provider,
                attempt_model,
                system_prompt,
                user_prompt,
                api_key,
                cfg,
                timeout,
                retry,
                retry_delay,
                with_reasoning=False,
            )
            result["misconfiguration_warning"] = msg

        if not result.get("failed"):
            _strip_internal(result)
            if attempt_model != requested:
                result["fallback_from"] = requested
                log.warning(
                    f"{provider} used FALLBACK model {attempt_model!r} because "
                    f"{requested!r} was unavailable (capacity). Review quality "
                    f"may be reduced."
                )
            return result

        # An exhausted account fails identically on every model in the chain,
        # so walking it just multiplies the same failure by three and reports
        # the last model's name as the one that broke.
        if (
            _is_capacity_error(result.get("_status"))
            and not result.get("_terminal")
            and attempt_model != chain[-1]
        ):
            log.warning(
                f"{provider} {attempt_model} unavailable (capacity). "
                f"Trying next fallback model."
            )
            continue
        _strip_internal(result)
        return result

    if result is not None:
        _strip_internal(result)
    return result
