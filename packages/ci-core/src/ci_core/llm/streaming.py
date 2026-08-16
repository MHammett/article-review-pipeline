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
Three layers, each with exactly one job:

  1. **First-byte allowance** (``stream_read_timeout``, socket read timeout via
     :func:`stream_timeout`) — covers everything before generation starts:
     queueing, a grounded model's web search, silent reasoning. Necessarily
     generous; sonar-reasoning-pro needs hundreds of seconds here.
  2. **Inter-chunk gap** (``stream_gap_timeout``, enforced by
     :func:`_iter_lines_with_gap`) — the liveness detector. Tight and roughly
     constant: a stream that has started emitting keeps emitting, so a long
     silence mid-response means the connection is dead, not slow.
  3. **Wall-clock backstop** (the pipeline's sliding-scale ``timeout_seconds``,
     enforced by ``ci_article_review.pipeline._run_with_timeout``) — "this call
     is alive but I'm not waiting any longer". Deliberately ignored here.

Layers 1 and 2 used to be a single value, which meant the stall detector had to
be sized for the *search* phase. That is how perplexity's read timeout reached
500s: each bump was chasing slow-but-alive calls, and the side effect was that a
genuinely dead sonar connection took over eight minutes to notice — the exact
thing a stall detector exists to prevent. Splitting them lets layer 1 stay
generous while layer 2 goes back to catching dead connections quickly.

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
import queue
import threading

import requests

log = logging.getLogger(__name__)

# Connection-establishment timeout (seconds). Small and constant — establishing
# the TCP/TLS connection has nothing to do with generation length.
DEFAULT_CONNECT_TIMEOUT = 30

# Default FIRST-BYTE allowance (seconds) when a model config does not set
# ``stream_read_timeout``. This covers everything that happens before generation
# starts: queueing, a grounded model's web search, and any reasoning the provider
# does without putting bytes on the wire. Grounded models (Gemini, Perplexity)
# override it upward because their search phase runs before the first token.
DEFAULT_FIRST_BYTE_TIMEOUT = 120

# Backwards-compatible alias. The old name described this value as an inter-token
# gap, but every value ever set for it was chosen to survive a pre-first-byte
# silence, which is what it now means explicitly.
DEFAULT_READ_TIMEOUT = DEFAULT_FIRST_BYTE_TIMEOUT

# Default inter-chunk GAP (seconds) — the liveness detector. Once a stream has
# produced its first chunk, a healthy provider keeps emitting: the worst gap ever
# observed here is ~8s (gpt-5.5 xhigh reasoning-summary deltas). Anything past a
# minute means the connection is dead, not slow.
#
# This is deliberately NOT sized to cover slow generation. Sizing the stall
# detector to survive slow-but-alive calls is what drove perplexity's value from
# 160 -> 280 -> 350 -> 500, and a 500s stall detector means an abandoned call
# holds a socket (and delays interpreter exit via the executor's atexit join —
# see ci_article_review.pipeline._run_with_timeout) for over eight minutes.
# Slow-but-alive is the wall-clock backstop's job; this one only catches dead.
DEFAULT_GAP_TIMEOUT = 60


def stream_timeout(cfg, default_read=DEFAULT_FIRST_BYTE_TIMEOUT):
    """Return the ``(connect, read)`` socket timeout tuple for a streaming request.

    ``read`` here is the FIRST-BYTE allowance, not the inter-chunk gap. It has to
    stay generous enough to survive a grounded model's search phase, so it cannot
    also serve as the stall detector — :func:`gap_timeout` supplies that, enforced
    independently of the socket (see :func:`_iter_lines_with_gap`).

    A per-model ``stream_read_timeout`` in the provider config overrides the
    default. The sliding-scale ``timeout_seconds`` is intentionally ignored here:
    it is the pipeline's wall-clock backstop, not a socket timeout.
    """
    read = (cfg or {}).get("stream_read_timeout") or default_read
    return (DEFAULT_CONNECT_TIMEOUT, read)


def gap_timeout(cfg, default_gap=DEFAULT_GAP_TIMEOUT):
    """Return the inter-chunk stall timeout (seconds) for a streaming request.

    Overridden per model with ``stream_gap_timeout``. Unlike the first-byte
    allowance this should rarely need raising: a provider that streams at all
    streams steadily, and a long gap mid-response means a dead connection.
    """
    return (cfg or {}).get("stream_gap_timeout") or default_gap


def _iter_lines_with_gap(resp, first_byte_seconds, gap_seconds):
    """Yield decoded lines, aborting if the stream goes quiet for too long.

    Two separate allowances: ``first_byte_seconds`` before the first chunk, and
    ``gap_seconds`` between every chunk after it.

    Why a reader thread rather than a clock check in the loop: ``iter_lines``
    blocks inside a socket read, so the consumer only regains control when a line
    arrives or the socket's own timeout fires. Since the socket timeout has to
    stay generous enough for the first byte, it cannot double as a tight stall
    detector. Reading in a daemon thread and consuming through a queue decouples
    the two — the consumer's ``Queue.get`` timeout enforces the gap regardless of
    what the socket is willing to wait for.

    On a stall the response is closed, which unblocks the reader's pending socket
    read so the thread exits promptly instead of lingering until the generous
    socket timeout expires. The thread is a daemon as a second line of defence:
    a stuck reader must never keep the interpreter alive.
    """
    q = queue.Queue()  # unbounded: a bounded queue could block the reader in
    # ``put`` after the consumer has given up and stopped draining.
    done = object()

    def _read():
        try:
            for line in resp.iter_lines(decode_unicode=True):
                q.put(line)
        except BaseException as exc:  # noqa: BLE001 — relayed to the consumer
            q.put(exc)
        finally:
            q.put(done)

    reader = threading.Thread(target=_read, name="sse-reader", daemon=True)
    reader.start()

    budget = first_byte_seconds
    started = False
    while True:
        try:
            item = q.get(timeout=budget)
        except queue.Empty:
            resp.close()
            phase = "mid-stream" if started else "before first chunk"
            raise requests.exceptions.ReadTimeout(
                f"SSE stream stalled {phase}: nothing received for {budget}s"
            )
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        started = True
        budget = gap_seconds
        yield item


def iter_sse_data(resp, first_byte=None, gap=None):
    """Yield the parsed JSON object from each ``data:`` line of an SSE response.

    Stops at the ``data: [DONE]`` sentinel. Non-``data:`` lines (``event:``,
    ``id:``, comments, blank keep-alives) and unparseable payloads are skipped.
    A stall while iterating propagates as a ``requests`` exception, which the
    calling adapter treats as a failed call.

    When ``gap`` is supplied the stream is read through :func:`_iter_lines_with_gap`,
    which enforces a tight inter-chunk stall timeout independently of the socket's
    (generous) first-byte allowance. With ``gap`` omitted the response is iterated
    directly and the socket timeout is the only bound — the pre-split behaviour,
    kept so tests and callers that hand in a plain stub response still work.

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
    if gap:
        lines = _iter_lines_with_gap(
            resp, first_byte or DEFAULT_FIRST_BYTE_TIMEOUT, gap
        )
    else:
        lines = resp.iter_lines(decode_unicode=True)
    for raw_line in lines:
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


def accumulate_chat_completions(resp, first_byte=None, gap=None):
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

    for obj in iter_sse_data(resp, first_byte=first_byte, gap=gap):
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


def accumulate_anthropic(resp, first_byte=None, gap=None):
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

    for obj in iter_sse_data(resp, first_byte=first_byte, gap=gap):
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


def accumulate_gemini(resp, first_byte=None, gap=None):
    """Accumulate a Gemini ``streamGenerateContent?alt=sse`` stream.

    Each ``data:`` chunk is a partial ``GenerateContentResponse``. Text parts are
    concatenated in order; parts flagged ``thought: true`` (internal reasoning) are
    skipped, matching the buffered path. ``usageMetadata`` is cumulative across
    chunks, so the last one seen wins. Also tracks the last non-empty
    ``finishReason`` seen across candidates — a ``MAX_TOKENS`` finish reason means
    generation was cut off before a complete JSON payload could be emitted, which
    is a distinct (and diagnosable) failure mode from the model genuinely
    returning malformed content. Returns the assembled text, the final
    ``usageMetadata`` dict, and ``finish_reason``.
    """
    parts = []
    usage = {}
    finish_reason = None

    for obj in iter_sse_data(resp, first_byte=first_byte, gap=gap):
        if obj.get("usageMetadata"):
            usage = obj["usageMetadata"]
        for cand in obj.get("candidates") or []:
            if cand.get("finishReason"):
                finish_reason = cand["finishReason"]
            for part in cand.get("content", {}).get("parts", []):
                if not part.get("thought") and "text" in part:
                    parts.append(part["text"])

    return {"content": "".join(parts), "usage": usage, "finish_reason": finish_reason}


def accumulate_openai_responses(resp, first_byte=None, gap=None):
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
    """
    parts = []
    reasoning_parts = []
    usage = {}

    for obj in iter_sse_data(resp, first_byte=first_byte, gap=gap):
        etype = obj.get("type")
        if etype == "response.output_text.delta":
            parts.append(obj.get("delta", "") or "")
        elif etype == "response.reasoning_summary_text.delta":
            reasoning_parts.append(obj.get("delta", "") or "")
        elif etype in ("response.completed", "response.incomplete"):
            usage = obj.get("response", {}).get("usage", {}) or usage

    return {
        "content": "".join(parts),
        "usage": usage,
        "reasoning_summary": "".join(reasoning_parts),
    }
