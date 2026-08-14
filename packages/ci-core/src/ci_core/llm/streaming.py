"""Shared Server-Sent-Events (SSE) streaming helpers for the review adapters.

Every review provider can return its answer as an SSE stream instead of one
buffered JSON body. Streaming changes what the HTTP read timeout *means*: with a
buffered POST the read timeout has to cover the model's entire compute time (a
gpt-5.5 xhigh fact-check buffers 16-30k reasoning+output tokens and sends nothing
for ~800s, so the timeout had to be ~819s). With ``stream=True`` the read timeout
is the gap between successive chunks, so it becomes a small, roughly constant
"max stall between tokens" value regardless of how long the full generation runs.

Timeout model
-------------
``requests`` accepts ``timeout=(connect, read)``. When ``stream=True`` the read
component applies to each socket read while iterating the body, i.e. it is the
inter-token gap. :func:`stream_timeout` builds that tuple. It deliberately does
NOT use the sliding-scale ``timeout_seconds`` value the pipeline computes — under
streaming that big number is a wall-clock *backstop* enforced by the pipeline's
per-task thread wrapper, not a socket timeout. A per-model ``stream_read_timeout``
overrides the adapter default if a grounded/search model needs a longer first-byte
allowance (search runs before the first token arrives).

Parsers
-------
``iter_sse_data`` yields the decoded JSON object from each ``data:`` line. The
per-provider accumulators sit on top of it:

  * :func:`accumulate_chat_completions` — OpenAI / Grok / Mistral / Perplexity
    (OpenAI-compatible ``choices[].delta`` deltas).
  * :func:`accumulate_anthropic`        — Anthropic ``content_block_delta`` events.
  * :func:`accumulate_gemini`           — Gemini ``streamGenerateContent?alt=sse``.
  * :func:`accumulate_openai_responses` — OpenAI Responses API typed events,
    including reasoning-summary deltas.

Each returns the fully assembled text plus the usage/grounding metadata the
adapters need, so the existing JSON parsing/validation runs unchanged on the
accumulated string.
"""

import json
import logging

log = logging.getLogger(__name__)

# Connection-establishment timeout (seconds). Small and constant — establishing
# the TCP/TLS connection has nothing to do with generation length.
DEFAULT_CONNECT_TIMEOUT = 30

# Default inter-token / read-gap timeout (seconds) when a model config does not
# set ``stream_read_timeout``. This is the maximum allowed stall between chunks,
# NOT the total generation budget. Grounded models (Gemini, Perplexity) run a web
# search before the first token, so they default higher to cover time-to-first-byte.
DEFAULT_READ_TIMEOUT = 120


def stream_timeout(cfg, default_read=DEFAULT_READ_TIMEOUT):
    """Return the ``(connect, read)`` timeout tuple for a streaming request.

    ``read`` is the inter-token gap, not the total generation time. A per-model
    ``stream_read_timeout`` in the provider config overrides the adapter default;
    the sliding-scale ``timeout_seconds`` is intentionally ignored here (it is the
    pipeline's wall-clock backstop, not a socket read timeout).
    """
    read = (cfg or {}).get("stream_read_timeout") or default_read
    return (DEFAULT_CONNECT_TIMEOUT, read)


def iter_sse_data(resp):
    """Yield the parsed JSON object from each ``data:`` line of an SSE response.

    Stops at the ``data: [DONE]`` sentinel. Non-``data:`` lines (``event:``,
    ``id:``, comments, blank keep-alives) and unparseable payloads are skipped.
    A read-gap timeout while iterating propagates out of ``iter_lines`` as a
    ``requests`` exception, which the calling adapter treats as a failed call.

    ``resp.encoding`` is forced to UTF-8 before any iteration starts. SSE
    (``text/event-stream``) is UTF-8 by spec, but none of the six providers'
    streaming responses send an explicit ``charset`` in their Content-Type
    header, so ``requests`` (``get_encoding_from_headers``) falls back to
    ISO-8859-1 for any ``text/*`` content type. ``iter_lines(decode_unicode=True)``
    then decodes every chunk through that wrong 1-byte-per-character encoding via
    ``stream_decode_response_unicode``, silently mangling every non-ASCII
    multi-byte character (e.g. a curly apostrophe, U+2019 / ``E2 80 99``) instead
    of raising — the classic mojibake failure mode. Setting the encoding here,
    before ``iter_lines`` touches the body, makes ``stream_decode_response_unicode``
    use a proper incremental UTF-8 decoder (``codecs.getincrementaldecoder``) that
    already buffers partial multi-byte sequences across chunk boundaries, so this
    one-line fix covers arbitrary chunk splits for all providers that route
    through this function.
    """
    resp.encoding = "utf-8"
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        # iter_lines may hand back bytes if decode_unicode is ignored by a mock.
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8", "replace")
        if not raw_line.startswith("data:"):
            continue
        payload = raw_line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def accumulate_chat_completions(resp):
    """Accumulate an OpenAI-compatible chat-completions stream.

    Shared by OpenAI, Grok, Mistral, and Perplexity. Returns a dict with the
    assembled ``content`` plus ``usage`` (from the final ``stream_options``
    include-usage chunk), ``finish_reason``, and Perplexity's ``citations`` /
    ``search_results`` when present (ignored by the other three).

    Mistral reasoning models stream ``delta.content`` as a list of typed chunks
    (``{"type": "thinking"...}``/``{"type": "text"...}``); text chunks only are
    kept, matching the buffered-path behavior.

    Some providers emit an in-band ``{"error": {...}}`` SSE event instead of
    (or in addition to) ``choices`` — e.g. a rate limit or content-policy
    rejection surfaced mid-stream rather than as an HTTP error status. That
    event carries no ``choices``, so it would otherwise silently produce an
    empty ``content`` with no usage, indistinguishable from a plain dropped
    connection. It's captured here as ``stream_error`` so callers can report
    the real cause instead of a generic "malformed JSON" message.
    """
    parts = []
    usage = {}
    finish_reason = None
    citations = []
    search_results = []
    stream_error = None

    for obj in iter_sse_data(resp):
        if obj.get("usage"):
            usage = obj["usage"]
        if obj.get("citations"):
            citations = obj["citations"]
        if obj.get("search_results"):
            search_results = obj["search_results"]
        if obj.get("error"):
            stream_error = obj["error"]
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, list):
                for chunk in piece:
                    if chunk.get("type") == "text":
                        parts.append(chunk.get("text", ""))
            elif piece:
                parts.append(piece)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    return {
        "content": "".join(parts),
        "usage": usage,
        "finish_reason": finish_reason,
        "citations": citations,
        "search_results": search_results,
        "stream_error": stream_error,
    }


def accumulate_anthropic(resp):
    """Accumulate an Anthropic ``/v1/messages`` SSE stream.

    Anthropic emits ``message_start`` (carries input-token usage),
    ``content_block_delta`` events (``text_delta`` for answer text, ``thinking_delta``
    for reasoning — only text is kept), and ``message_delta`` (carries the final
    output-token usage and ``stop_reason``). Returns the assembled text plus a
    ``usage`` dict shaped like the buffered response (``input_tokens`` /
    ``output_tokens``) and the ``stop_reason``.
    """
    parts = []
    usage = {}
    stop_reason = None

    for obj in iter_sse_data(resp):
        etype = obj.get("type")
        if etype == "message_start":
            msg_usage = obj.get("message", {}).get("usage", {})
            if msg_usage:
                usage.update(
                    {
                        "input_tokens": msg_usage.get("input_tokens"),
                        "output_tokens": msg_usage.get("output_tokens"),
                    }
                )
                # Prompt caching moves most of the input out of input_tokens
                # into these two, which are disjoint from it and from each
                # other. Dropping them here made a cached call look almost
                # free — 20 prompt tokens for a 4,800-token system prompt.
                for key in (
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ):
                    if msg_usage.get(key) is not None:
                        usage[key] = msg_usage[key]
        elif etype == "content_block_delta":
            delta = obj.get("delta", {})
            if delta.get("type") == "text_delta":
                parts.append(delta.get("text", ""))
        elif etype == "message_delta":
            if obj.get("delta", {}).get("stop_reason"):
                stop_reason = obj["delta"]["stop_reason"]
            out = obj.get("usage", {}).get("output_tokens")
            if out is not None:
                usage["output_tokens"] = out

    return {"content": "".join(parts), "usage": usage, "stop_reason": stop_reason}


def accumulate_gemini(resp):
    """Accumulate a Gemini ``streamGenerateContent?alt=sse`` stream.

    Each ``data:`` chunk is a partial ``GenerateContentResponse``. Text parts are
    concatenated in order; parts flagged ``thought: true`` (internal reasoning) are
    skipped, matching the buffered path. ``usageMetadata`` is cumulative across
    chunks, so the last one seen wins. Also tracks the last non-empty
    ``finishReason`` seen across candidates — a ``MAX_TOKENS`` finish reason means
    generation was cut off before a complete JSON payload could be emitted, which
    is a distinct (and diagnosable) failure mode from the model genuinely
    returning malformed content.

    Also collects ``candidates[].groundingMetadata.groundingChunks`` — the web
    sources the model's live Google Search actually returned. Every grounded
    Gemini call has been producing these and nothing read them, which left
    Perplexity as the pipeline's sole supplier of grounded citation URLs. Chunks
    arrive across multiple SSE chunks and repeat, so they are deduplicated on
    ``uri`` with first-seen order preserved.

    Note the ``uri`` values are ``vertexaisearch.cloud.google.com/
    grounding-api-redirect/...`` wrappers rather than source URLs, and they
    expire after roughly 30 days. They are returned raw here — resolving them is
    network I/O and does not belong on the streaming path; see
    :func:`ci_core.llm.adapters.gemini.resolve_grounding_urls`.

    Returns the assembled text, the final ``usageMetadata`` dict,
    ``finish_reason``, and ``grounding_chunks`` (``[{"uri", "title"}, ...]``).
    """
    parts = []
    usage = {}
    finish_reason = None
    grounding_chunks = []
    seen_uris = set()

    for obj in iter_sse_data(resp):
        if obj.get("usageMetadata"):
            usage = obj["usageMetadata"]
        for cand in obj.get("candidates") or []:
            if cand.get("finishReason"):
                finish_reason = cand["finishReason"]
            for chunk in (cand.get("groundingMetadata") or {}).get(
                "groundingChunks"
            ) or []:
                web = chunk.get("web") or {}
                uri = web.get("uri")
                if isinstance(uri, str) and uri not in seen_uris:
                    seen_uris.add(uri)
                    grounding_chunks.append(
                        {"uri": uri, "title": web.get("title") or ""}
                    )
            for part in cand.get("content", {}).get("parts", []):
                if not part.get("thought") and "text" in part:
                    parts.append(part["text"])

    return {
        "content": "".join(parts),
        "usage": usage,
        "finish_reason": finish_reason,
        "grounding_chunks": grounding_chunks,
    }


def accumulate_openai_responses(resp):
    """Accumulate an OpenAI Responses API (``/v1/responses``) SSE stream.

    The Responses API streams typed events: ``response.output_text.delta`` carries
    answer text in ``delta``; ``response.completed`` carries the final ``response``
    object including ``usage``. Returns the assembled text and the usage dict
    (``input_tokens`` / ``output_tokens`` shape).

    Reasoning models (``reasoning.summary`` requested in the payload) also emit
    ``response.reasoning_summary_text.added`` / ``.delta`` / ``.done`` while the
    model is still "thinking" — before this, reasoning models sent zero bytes
    during that phase, which is what forced the inter-token read-gap timeout up
    to 200-300s for high/xhigh effort (see ci_core/llm/adapters/openai.py). These
    events carry no answer content, but reading them off the socket is what
    resets the read-gap timer, so the summary text is captured (for
    debug/logging) rather than discarded outright. Any other typed event
    (``response.created``, ``response.in_progress``, ...) is ignored — not
    crashing on it is enough for it to reset the timer.

    When the ``web_search_preview`` tool is enabled, cited sources arrive as
    ``response.output_text.annotation.added`` events carrying ``{"url",
    "title"}``, restated on the final response object. They are collected into
    ``citations`` — the same key Perplexity uses — and unlike Gemini's grounding
    chunks they are real source URLs needing no redirect resolution.

    **This is inert under ci-article-review's own prompts, by design of those
    prompts.** Measured 2026-08-12: the search itself always runs, but the model
    only emits annotations when it writes prose with inline citations. Every
    review domain demands JSON-only output, so annotations come back empty every
    time. What web search actually buys that pipeline is a *better-sourced*
    answer — the ``source`` field the model fills becomes a URL it fetched this
    run rather than one recalled from training data — which reaches the resolver
    through the existing ``known_url`` path, not through here. The capture below
    is kept because it is correct and free, and it activates for any caller whose
    prompt does ask for prose. It is not a citation source for this pipeline
    today; do not count it as one.
    """
    parts = []
    reasoning_parts = []
    usage = {}
    citations = []
    seen_urls = set()

    def _take_annotation(annotation):
        if not isinstance(annotation, dict):
            return
        url = annotation.get("url")
        if isinstance(url, str) and url and url not in seen_urls:
            seen_urls.add(url)
            citations.append(url)

    for obj in iter_sse_data(resp):
        etype = obj.get("type")
        if etype == "response.output_text.delta":
            parts.append(obj.get("delta", "") or "")
        elif etype == "response.reasoning_summary_text.delta":
            reasoning_parts.append(obj.get("delta", "") or "")
        elif etype == "response.output_text.annotation.added":
            _take_annotation(obj.get("annotation"))
        elif etype in ("response.completed", "response.incomplete"):
            response = obj.get("response") or {}
            usage = response.get("usage", {}) or usage
            # Restated on the final object; harmless to re-read thanks to seen_urls.
            for item in response.get("output") or []:
                for content in item.get("content") or []:
                    for annotation in content.get("annotations") or []:
                        _take_annotation(annotation)

    return {
        "content": "".join(parts),
        "usage": usage,
        "reasoning_summary": "".join(reasoning_parts),
        "citations": citations,
    }
