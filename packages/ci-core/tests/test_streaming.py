"""Regression tests for shared SSE decoding in ci_core/llm/streaming.py.

These use a *real* ``requests.Response`` (not a string-level mock) with a fake
``.raw`` object, so the test exercises the actual ``requests`` encoding-detection
and incremental-decode machinery (``get_encoding_from_headers`` /
``stream_decode_response_unicode``) that produced the mojibake bug, rather than
mocking ``iter_lines`` at the already-decoded-string level.
"""

import json

import pytest
from unittest.mock import patch, MagicMock

import requests

from ci_core.llm import streaming


class _FakeRaw:
    """Stands in for the urllib3 HTTPResponse ``requests`` reads from.

    ``Response.iter_content`` calls ``self.raw.stream(chunk_size, decode_content=True)``
    when the attribute exists, and yields whatever byte chunks it produces
    verbatim (chunk_size is advisory, not a hard split point) -- exactly like a
    real socket read would.
    """

    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, chunk_size=1, decode_content=True):
        yield from self._chunks


def _sse_response_with_split_multibyte_char(chunks, content_type="text/event-stream"):
    """A real ``requests.Response`` whose body bytes arrive as the given chunks.

    ``encoding`` is set the way ``requests``' own transport adapter
    (``HTTPAdapter.build_response``) sets it in production: from
    ``get_encoding_from_headers`` on the response's Content-Type. Providers'
    streaming endpoints send ``text/event-stream`` with no explicit ``charset``,
    which that function maps to ISO-8859-1 (any ``text/*`` type with no charset)
    -- the actual bug, reproduced here rather than assumed.
    """
    resp = requests.Response()
    resp.status_code = 200
    resp.headers = requests.structures.CaseInsensitiveDict(
        {"Content-Type": content_type}
    )
    resp.encoding = requests.utils.get_encoding_from_headers(resp.headers)
    resp.raw = _FakeRaw(chunks)
    return resp


def _split_utf8_line_across_chunks(line, split_after_bytes):
    """Encode ``line`` as UTF-8 and split it into two chunks at a byte offset."""
    raw = line.encode("utf-8")
    return [raw[:split_after_bytes], raw[split_after_bytes:]]


class TestIterSseDataUtf8Decoding:
    """Direct tests of the shared ``iter_sse_data`` used by all six adapters."""

    def test_curly_apostrophe_split_across_chunk_boundary_decodes_correctly(self):
        # U+2019 RIGHT SINGLE QUOTATION MARK -> UTF-8 bytes E2 80 99. Split the
        # stream so the E2 lands in one chunk and 80 99 land in the next --
        # exactly the failure mode observed in the captured raw_excerpt
        # ("utility\x80\x99s knowledge" instead of "utility's knowledge").
        line = "data: " + json.dumps(
            {"choices": [{"delta": {"content": "utility’s knowledge"}}]},
            ensure_ascii=False,
        )
        raw = line.encode("utf-8")
        split_at = raw.index(b"\xe2\x80\x99") + 1  # split right after the E2 byte
        chunks = _split_utf8_line_across_chunks(line, split_at)
        resp = _sse_response_with_split_multibyte_char(chunks)

        assert (
            resp.encoding == "ISO-8859-1"
        )  # confirms the bug is reproduced, not assumed

        events = list(streaming.iter_sse_data(resp))

        assert len(events) == 1
        content = events[0]["choices"][0]["delta"]["content"]
        assert content == "utility’s knowledge"
        assert "\x80" not in content
        assert "\x99" not in content

    def test_multiple_multibyte_chars_split_at_different_offsets(self):
        # Two separate curly apostrophes, split at different byte offsets within
        # their own 3-byte UTF-8 sequences, to make sure the fix isn't tied to one
        # specific split point.
        text = "Uptime’s 2025 report shows the utility’s growth"
        line = "data: " + json.dumps(
            {"choices": [{"delta": {"content": text}}]}, ensure_ascii=False
        )
        raw = line.encode("utf-8")
        first_seq = raw.index(b"\xe2\x80\x99")
        second_seq = raw.index(b"\xe2\x80\x99", first_seq + 3)
        chunks = [
            raw[: first_seq + 2],  # split inside the first sequence (E2 80 | 99)
            raw[first_seq + 2 : second_seq + 1],  # split inside the second (E2 | 80 99)
            raw[second_seq + 1 :],
        ]
        resp = _sse_response_with_split_multibyte_char(chunks)

        events = list(streaming.iter_sse_data(resp))

        content = events[0]["choices"][0]["delta"]["content"]
        assert content == text

    def test_intact_multibyte_char_in_a_single_chunk_still_decodes_correctly(self):
        # Sanity check: the fix must not regress the non-split case.
        text = "the utility’s knowledge base"
        line = "data: " + json.dumps(
            {"choices": [{"delta": {"content": text}}]}, ensure_ascii=False
        )
        resp = _sse_response_with_split_multibyte_char([line.encode("utf-8")])

        events = list(streaming.iter_sse_data(resp))

        assert events[0]["choices"][0]["delta"]["content"] == text

    def test_forces_utf8_even_when_charset_header_present(self):
        # A provider that *does* send an explicit (wrong) charset should still be
        # decoded as UTF-8 -- SSE is UTF-8 by spec regardless of what a
        # misconfigured Content-Type claims.
        text = "the utility’s knowledge base"
        line = "data: " + json.dumps(
            {"choices": [{"delta": {"content": text}}]}, ensure_ascii=False
        )
        resp = _sse_response_with_split_multibyte_char(
            [line.encode("utf-8")], content_type="text/event-stream; charset=iso-8859-1"
        )

        events = list(streaming.iter_sse_data(resp))

        assert events[0]["choices"][0]["delta"]["content"] == text


class TestPerplexityAdapterEndToEndDecoding:
    """Confirms the fix reaches all the way through the real Perplexity adapter,
    not just the shared helper in isolation -- this is the provider where the
    bug was originally observed via the raw_excerpt diagnostics from PR #21.
    """

    def test_call_decodes_split_curly_apostrophe_correctly(self):
        from ci_core.llm.adapters import perplexity as pplx

        answer = json.dumps(
            {"summary": "The utility’s 2025 Uptime’s report", "claims": []},
            ensure_ascii=False,
        )
        content_line = "data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {"content": answer}}]},
            ensure_ascii=False,
        )
        raw = content_line.encode("utf-8")
        split_at = raw.index(b"\xe2\x80\x99") + 1
        final_line = "data: " + json.dumps(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )
        done_line = "data: [DONE]"

        chunks = [
            raw[:split_at],
            raw[split_at:]
            + b"\n"
            + final_line.encode("utf-8")
            + b"\n"
            + done_line.encode("utf-8"),
        ]
        resp = _sse_response_with_split_multibyte_char(chunks)
        resp.raise_for_status = MagicMock()
        resp.close = MagicMock()

        with patch(
            "ci_core.llm.adapters.perplexity.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = resp
            result = pplx.call("system", "user", "key", retry=False)

        assert not result.get("failed"), result.get("error")
        summary = result["data"]["summary"]
        assert "\x80" not in summary
        assert "’" in summary


class _StallingRaw:
    """A body that yields some chunks and then blocks forever.

    Models the failure the gap timeout exists to catch: the connection is open
    and the socket read is still pending, but no more bytes are coming. A real
    socket would eventually raise on its own read timeout -- which is exactly the
    generous first-byte value we do NOT want to wait for mid-stream.
    """

    def __init__(self, chunks, released):
        self._chunks = chunks
        self._released = released
        self.stream_exited_early = False

    def stream(self, chunk_size=1, decode_content=True):
        yield from self._chunks
        # Block until the consumer gives up and closes the response.
        if self._released.wait(timeout=10):
            self.stream_exited_early = True
            raise OSError("socket closed by consumer")


def _stalling_response(chunks, released):
    resp = requests.Response()
    resp.status_code = 200
    resp.headers = requests.structures.CaseInsensitiveDict(
        {"Content-Type": "text/event-stream"}
    )
    resp.encoding = "utf-8"
    raw = _StallingRaw(chunks, released)
    resp.raw = raw
    # ``Response.close`` calls ``raw.close()`` and then ``raw.release_conn()``
    # when present. Both unblock the reader here, the way closing a real socket
    # would release a thread parked in a pending read.
    raw.close = released.set
    raw.release_conn = released.set
    return resp, raw


class TestGapTimeoutResolution:
    """``stream_read_timeout`` and ``stream_gap_timeout`` are separate knobs."""

    def test_gap_defaults_when_unset(self):
        assert streaming.gap_timeout(None) == streaming.DEFAULT_GAP_TIMEOUT
        assert streaming.gap_timeout({}) == streaming.DEFAULT_GAP_TIMEOUT

    def test_gap_is_overridable_per_model(self):
        assert streaming.gap_timeout({"stream_gap_timeout": 45}) == 45

    def test_stream_read_timeout_does_not_change_the_gap(self):
        """The two are independent: a grounded model needs a long first-byte
        allowance without loosening its stall detector."""
        cfg = {"stream_read_timeout": 500}
        assert streaming.stream_timeout(cfg)[1] == 500
        assert streaming.gap_timeout(cfg) == streaming.DEFAULT_GAP_TIMEOUT

    def test_socket_timeout_carries_the_first_byte_allowance(self):
        connect, read = streaming.stream_timeout({"stream_read_timeout": 500})
        assert connect == streaming.DEFAULT_CONNECT_TIMEOUT
        assert read == 500


class TestInterChunkGapEnforcement:
    def test_healthy_stream_is_unaffected(self):
        body = b'data: {"n": 1}\n\ndata: {"n": 2}\n\ndata: [DONE]\n\n'
        resp = _sse_response_with_split_multibyte_char([body])
        got = list(streaming.iter_sse_data(resp, first_byte=30, gap=30))
        assert got == [{"n": 1}, {"n": 2}]

    def test_stall_after_first_chunk_raises_before_the_socket_would(self):
        """The whole point of the split: a mid-stream stall is caught on the gap
        (short), not on the first-byte allowance (long)."""
        import threading
        import time as _time

        released = threading.Event()
        resp, raw = _stalling_response([b'data: {"n": 1}\n\n'], released)
        t0 = _time.monotonic()
        with pytest.raises(requests.exceptions.ReadTimeout) as exc:
            # first_byte is deliberately huge; only the gap should govern here.
            list(streaming.iter_sse_data(resp, first_byte=600, gap=1))
        elapsed = _time.monotonic() - t0
        assert "mid-stream" in str(exc.value)
        assert elapsed < 10, f"gap timeout did not fire promptly ({elapsed:.1f}s)"

    def test_stall_before_first_chunk_uses_the_first_byte_allowance(self):
        """A grounded model's search phase must not trip the (short) gap."""
        import threading
        import time as _time

        released = threading.Event()
        resp, raw = _stalling_response([], released)
        t0 = _time.monotonic()
        with pytest.raises(requests.exceptions.ReadTimeout) as exc:
            list(streaming.iter_sse_data(resp, first_byte=1, gap=600))
        elapsed = _time.monotonic() - t0
        assert "before first chunk" in str(exc.value)
        assert elapsed < 10

    def test_stalled_response_is_closed_so_the_reader_thread_exits(self):
        """A dead call must not keep holding a socket -- the reason the stall
        detector exists at all (see pipeline._run_with_timeout)."""
        import threading

        released = threading.Event()
        resp, _raw = _stalling_response([b'data: {"n": 1}\n\n'], released)
        with pytest.raises(requests.exceptions.ReadTimeout):
            list(streaming.iter_sse_data(resp, first_byte=600, gap=1))
        assert released.wait(timeout=5), "response was not closed on stall"

    def test_omitting_gap_keeps_the_pre_split_behaviour(self):
        """Callers that pass no gap iterate the response directly."""
        body = b'data: {"n": 1}\n\ndata: [DONE]\n\n'
        resp = _sse_response_with_split_multibyte_char([body])
        assert list(streaming.iter_sse_data(resp)) == [{"n": 1}]
