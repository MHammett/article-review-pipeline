"""Regression tests for shared SSE decoding in adapters/review/streaming.py.

These use a *real* ``requests.Response`` (not a string-level mock) with a fake
``.raw`` object, so the test exercises the actual ``requests`` encoding-detection
and incremental-decode machinery (``get_encoding_from_headers`` /
``stream_decode_response_unicode``) that produced the mojibake bug, rather than
mocking ``iter_lines`` at the already-decoded-string level.
"""

import json
from unittest.mock import patch, MagicMock

import requests

from ci_article_review.adapters.review import streaming


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
        from ci_article_review.adapters.review import perplexity as pplx

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
            "ci_article_review.adapters.review.perplexity.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = resp
            result = pplx.call("system", "user", "key", retry=False)

        assert not result.get("failed"), result.get("error")
        summary = result["data"]["summary"]
        assert "\x80" not in summary
        assert "’" in summary
