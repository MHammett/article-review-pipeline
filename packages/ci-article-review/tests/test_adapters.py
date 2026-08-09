"""Unit tests for adapter modules."""

import os
import json
import pytest
import requests
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# SSE streaming mock helpers
# ---------------------------------------------------------------------------
# The review adapters POST with stream=True and consume the response via
# resp.iter_lines(). These helpers build mock responses that yield provider-shaped
# SSE `data:` lines so the accumulators in adapters/review/streaming.py parse them
# exactly as they would a live stream.


def _sse_mock(lines, status=200):
    """A mock streaming response whose iter_lines() yields the given SSE lines."""
    mock = MagicMock()
    mock.status_code = status
    mock.raise_for_status = MagicMock()
    mock.close = MagicMock()
    # Return a fresh iterator each call so a retry that re-reads still works.
    mock.iter_lines.side_effect = lambda *a, **k: iter(list(lines))
    return mock


def _split(text, n=3):
    """Split text into roughly n non-empty chunks to exercise delta accumulation."""
    if not text:
        return [""]
    size = max(1, -(-len(text) // n))
    return [text[i : i + size] for i in range(0, len(text), size)]


def _sse_chat_lines(content, usage=None, finish_reason="stop", extras=None, n=3):
    """OpenAI-compatible chat-completions stream (OpenAI/Grok/Mistral/Perplexity)."""
    text = content if isinstance(content, str) else json.dumps(content)
    lines = []
    for piece in _split(text, n):
        lines.append(
            "data: "
            + json.dumps({"choices": [{"index": 0, "delta": {"content": piece}}]})
        )
        lines.append("")
    final = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
    final["usage"] = (
        usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 50}
    )
    if extras:
        final.update(extras)
    lines.append("data: " + json.dumps(final))
    lines.append("data: [DONE]")
    return lines


def _sse_responses_lines(content, usage=None, reasoning_deltas=None, n=3):
    """OpenAI Responses API stream (openai.com default + web_search paths).

    ``reasoning_deltas``, if given, emits ``response.reasoning_summary_text.delta``
    events before the answer deltas — reasoning models stream these during the
    silent "thinking" phase, ahead of any output text.
    """
    text = content if isinstance(content, str) else json.dumps(content)
    lines = []
    for piece in reasoning_deltas or []:
        lines.append(
            "data: "
            + json.dumps(
                {"type": "response.reasoning_summary_text.delta", "delta": piece}
            )
        )
        lines.append("")
    for piece in _split(text, n):
        lines.append(
            "data: "
            + json.dumps({"type": "response.output_text.delta", "delta": piece})
        )
        lines.append("")
    lines.append(
        "data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "usage": usage
                    if usage is not None
                    else {"input_tokens": 100, "output_tokens": 50}
                },
            }
        )
    )
    lines.append("data: [DONE]")
    return lines


def _sse_anthropic_lines(content, input_tokens=100, output_tokens=50, n=3):
    """Anthropic /v1/messages stream."""
    text = content if isinstance(content, str) else json.dumps(content)
    lines = [
        "event: message_start",
        "data: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}
                },
            }
        ),
        "",
        "data: "
        + json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        "",
    ]
    for piece in _split(text, n):
        lines.append(
            "data: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": piece},
                }
            )
        )
        lines.append("")
    lines.append(
        "data: "
        + json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": output_tokens},
            }
        )
    )
    lines.append("data: " + json.dumps({"type": "message_stop"}))
    return lines


def _sse_gemini_lines(content, prompt_tokens=120, completion_tokens=60, n=3):
    """Gemini streamGenerateContent?alt=sse stream."""
    text = content if isinstance(content, str) else json.dumps(content)
    lines = []
    for piece in _split(text, n):
        lines.append(
            "data: "
            + json.dumps({"candidates": [{"content": {"parts": [{"text": piece}]}}]})
        )
        lines.append("")
    lines.append(
        "data: "
        + json.dumps(
            {
                "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": prompt_tokens,
                    "candidatesTokenCount": completion_tokens,
                },
            }
        )
    )
    return lines


# ---------------------------------------------------------------------------
# LanguageTool adapter
# ---------------------------------------------------------------------------


class TestLanguageTool:
    def _make_match(self, offset, length, category_id, replacement, rule_id="RULE"):
        return {
            "offset": offset,
            "length": length,
            "message": "Test message",
            "rule": {"id": rule_id, "category": {"id": category_id}},
            "replacements": [{"value": replacement}],
            "context": {"text": "context text"},
        }

    def test_apply_corrections_auto_apply(self):
        from ci_article_review.adapters.grammar.languagetool import apply_corrections

        text = "Teh quick brown fox"
        matches = [self._make_match(0, 3, "TYPOS", "The")]
        corrected, log = apply_corrections(text, matches, {"TYPOS"}, set())
        assert corrected == "The quick brown fox"
        assert len(log) == 1
        assert log[0]["original"] == "Teh"
        assert log[0]["replacement"] == "The"

    def test_apply_corrections_suppressed(self):
        from ci_article_review.adapters.grammar.languagetool import apply_corrections

        text = "Running fast."
        matches = [self._make_match(0, 7, "SENTENCE_FRAGMENT", "Run")]
        corrected, log = apply_corrections(
            text, matches, {"TYPOS"}, {"SENTENCE_FRAGMENT"}
        )
        assert corrected == text
        assert len(log) == 0

    def test_apply_corrections_not_auto_apply_category(self):
        from ci_article_review.adapters.grammar.languagetool import apply_corrections

        text = "Some text here."
        matches = [self._make_match(0, 4, "STYLE", "Different")]
        corrected, log = apply_corrections(text, matches, {"TYPOS"}, set())
        assert corrected == text
        assert len(log) == 0

    def test_apply_corrections_multiple_reverse_order(self):
        from ci_article_review.adapters.grammar.languagetool import apply_corrections

        text = "Teh cat sat on teh mat"
        matches = [
            self._make_match(0, 3, "TYPOS", "The"),
            self._make_match(15, 3, "TYPOS", "the"),
        ]
        corrected, log = apply_corrections(text, matches, {"TYPOS"}, set())
        assert "Teh" not in corrected
        assert "teh" not in corrected

    def test_run_languagetool_failure(self):
        from ci_article_review.adapters.grammar.languagetool import run

        lt_config = {"auto_apply": ["TYPOS"], "flag_for_review": [], "suppress": []}
        with patch(
            "ci_article_review.adapters.grammar.languagetool.check_text",
            side_effect=Exception("API down"),
        ):
            result = run("Test text", lt_config, "user@example.com", "key", retry=False)
        assert result["failed"] is True
        assert result["corrected_text"] == "Test text"
        assert "elapsed_seconds" in result

    def test_run_languagetool_success(self):
        from ci_article_review.adapters.grammar.languagetool import run

        lt_config = {
            "auto_apply": ["TYPOS"],
            "flag_for_review": ["STYLE"],
            "suppress": [],
        }
        mock_response = {
            "matches": [
                {
                    "offset": 0,
                    "length": 3,
                    "message": "Spelling",
                    "rule": {"id": "SPELL", "category": {"id": "TYPOS"}},
                    "replacements": [{"value": "The"}],
                    "context": {"text": "Teh quick"},
                }
            ]
        }
        with patch(
            "ci_article_review.adapters.grammar.languagetool.check_text",
            return_value=mock_response,
        ):
            result = run("Teh quick", lt_config, "user@example.com", "key")
        assert result["failed"] is False
        assert result["corrected_text"] == "The quick"
        assert len(result["change_log"]) == 1
        assert "elapsed_seconds" in result


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class TestOpenAI:
    def _mock_response(self, content_dict, status=200, usage=None):
        return _sse_mock(_sse_responses_lines(content_dict, usage=usage), status=status)

    def test_successful_call(self):
        from ci_article_review.adapters.review import openai as oai

        content = {
            "flags": [
                {"passage": "test", "problem": "hedging", "suggested_rewrite": "direct"}
            ],
            "low_confidence": [],
        }
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = oai.call("system", "user", "key")
        assert result["failed"] is False
        assert result["data"]["flags"][0]["passage"] == "test"
        assert result["tokens"]["prompt"] == 100
        assert "elapsed_seconds" in result
        # Primary path is the Responses API — not Chat Completions.
        assert "responses" in mock_session.post.call_args.args[
            0
        ] or "responses" in mock_session.post.call_args.kwargs.get("url", "")
        body = mock_session.post.call_args.kwargs["json"]
        assert "instructions" in body and "input" in body
        assert "messages" not in body

    def test_streamed_response_accumulated_and_parsed(self):
        # The JSON arrives split across many small deltas; the accumulator must
        # reassemble it before parsing.
        from ci_article_review.adapters.review import openai as oai

        content = {
            "flags": [{"passage": "p", "problem": "x", "suggested_rewrite": "y"}],
            "low_confidence": ["a", "b"],
        }
        lines = _sse_responses_lines(content, n=12)  # many tiny chunks
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = oai.call("system", "user", "key")
        assert result["failed"] is False
        assert result["data"] == content
        # stream=True must be sent both to requests and in the JSON body.
        assert mock_session.post.call_args.kwargs["stream"] is True
        assert mock_session.post.call_args.kwargs["json"]["stream"] is True

    def test_reasoning_summary_deltas_precede_answer_without_corrupting_output(self):
        # Reasoning-summary text deltas stream ahead of the answer deltas during
        # the model's silent "thinking" phase; they must be consumed (so they
        # reset the read-gap timer) without leaking into the assembled JSON.
        from ci_article_review.adapters.review import openai as oai

        content = {"flags": [], "low_confidence": []}
        lines = _sse_responses_lines(
            content, reasoning_deltas=["Weigh", "ing the ", "claims..."]
        )
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = oai.call(
                "system",
                "user",
                "key",
                provider_config={"model": "gpt-5.5", "reasoning_effort": "xhigh"},
            )
        assert result["failed"] is False
        assert result["data"] == content

    def test_usage_tokens_captured_from_final_chunk(self):
        from ci_article_review.adapters.review import openai as oai

        content = {"flags": [], "low_confidence": []}
        usage = {"input_tokens": 4321, "output_tokens": 876}
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content, usage=usage)
            result = oai.call("system", "user", "key")
        assert result["tokens"]["prompt"] == 4321
        assert result["tokens"]["completion"] == 876

    def test_inter_token_stall_triggers_read_timeout(self):
        # A stall between tokens surfaces as a requests read timeout while iterating
        # the stream; the adapter must report a failed call, not hang or crash.
        from ci_article_review.adapters.review import openai as oai

        def stalling_lines(*a, **k):
            yield "data: " + json.dumps(
                {"type": "response.reasoning_summary_text.delta", "delta": "..."}
            )
            raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=120)")

        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.close = MagicMock()
        mock.iter_lines.side_effect = stalling_lines
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = mock
            result = oai.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert (
            "timed out" in result["error"].lower()
            or "timeout" in result["error"].lower()
        )

    def test_failed_call(self):
        from ci_article_review.adapters.review import openai as oai

        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("Connection error")
            result = oai.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert "elapsed_seconds" in result

    def test_malformed_json(self):
        from ci_article_review.adapters.review import openai as oai

        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(
                _sse_responses_lines("not json at all")
            )
            result = oai.call("system", "user", "key")
        assert result["failed"] is True
        assert result["raw"] == "not json at all"

    def test_model_override(self):
        from ci_article_review.adapters.review import openai as oai

        content = {"flags": [], "low_confidence": []}
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = oai.call("system", "user", "key", model="gpt-4-turbo")
        assert result["model"] == "gpt-4-turbo"

    def test_reasoning_payload_uses_reasoning_object(self):
        # Responses API nests effort/summary under "reasoning" — must not send
        # Chat Completions' old top-level "reasoning_effort" field, and
        # temperature (incompatible with reasoning mode) must be omitted.
        from ci_article_review.adapters.review import openai as oai

        content = {"flags": [], "low_confidence": []}
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            oai.call(
                "system",
                "user",
                "key",
                provider_config={"model": "gpt-5.5", "reasoning_effort": "xhigh"},
            )
        body = mock_session.post.call_args.kwargs["json"]
        assert body["reasoning"] == {"effort": "xhigh", "summary": "auto"}
        assert "reasoning_effort" not in body
        assert "temperature" not in body

    def test_no_reasoning_effort_omits_reasoning_key(self):
        from ci_article_review.adapters.review import openai as oai

        content = {"flags": [], "low_confidence": []}
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            oai.call("system", "user", "key", provider_config={"model": "gpt-5.4"})
        body = mock_session.post.call_args.kwargs["json"]
        assert "reasoning" not in body
        assert body["temperature"] == 0.2

    def test_read_gap_timeout_constant_not_derived_from_timeout_seconds(self):
        # Under streaming, the socket read timeout is the inter-token gap — a small
        # constant. The big sliding-scale timeout_seconds must NOT inflate it (that
        # value is now only the pipeline's wall-clock backstop).
        from ci_article_review.adapters.review import openai as oai

        content = {"flags": [], "low_confidence": []}
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            oai.call(
                "system",
                "user",
                "key",
                provider_config={"model": "gpt-5.5", "timeout_seconds": 819},
            )
        connect, read = mock_session.post.call_args.kwargs["timeout"]
        assert (
            read == 120
        )  # constant inter-token gap, independent of timeout_seconds=819

    def test_default_timeout_when_unset(self):
        from ci_article_review.adapters.review import openai as oai

        content = {"flags": [], "low_confidence": []}
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            oai.call("system", "user", "key", provider_config={"model": "gpt-5.4"})
        assert mock_session.post.call_args.kwargs["timeout"] == (30, 120)

    def test_stream_read_timeout_override(self):
        # A grounded/slow model can widen the inter-token allowance explicitly.
        from ci_article_review.adapters.review import openai as oai

        content = {"flags": [], "low_confidence": []}
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            oai.call(
                "system",
                "user",
                "key",
                provider_config={"model": "gpt-5.4", "stream_read_timeout": 200},
            )
        assert mock_session.post.call_args.kwargs["timeout"] == (30, 200)

    def test_web_search_responses_api_streamed(self):
        # web_search: true → Responses API, which streams typed events
        # (response.output_text.delta) and reports usage on response.completed.
        from ci_article_review.adapters.review import openai as oai

        content = {"confirmed": ["claim"], "outdated": []}
        text = json.dumps(content)
        lines = []
        for piece in _split(text, 4):
            lines.append(
                "data: "
                + json.dumps({"type": "response.output_text.delta", "delta": piece})
            )
            lines.append("")
        lines.append(
            "data: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 321, "output_tokens": 123}},
                }
            )
        )
        lines.append("data: [DONE]")
        with patch(
            "ci_article_review.adapters.review.openai.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = oai.call(
                "system",
                "user",
                "key",
                provider_config={"model": "gpt-5.4", "web_search": True},
            )
        assert result["failed"] is False
        assert result["data"] == content
        assert result["grounding_available"] is True
        assert result["model"] == "gpt-5.4+search"
        assert result["tokens"]["prompt"] == 321
        assert result["tokens"]["completion"] == 123
        # Responses API endpoint + stream flag.
        assert "responses" in mock_session.post.call_args.args[
            0
        ] or "responses" in mock_session.post.call_args.kwargs.get("url", "")
        assert mock_session.post.call_args.kwargs["json"]["stream"] is True


# ---------------------------------------------------------------------------
# Mistral adapter
# ---------------------------------------------------------------------------


class TestMistral:
    def _mock_response(self, content_dict):
        return _sse_mock(
            _sse_chat_lines(
                content_dict, usage={"prompt_tokens": 80, "completion_tokens": 40}
            )
        )

    def test_reasoning_chunked_content_accumulated(self):
        # mistral-medium-3-5 streams delta.content as typed chunks (thinking + text);
        # only the text chunks form the answer JSON.
        from ci_article_review.adapters.review import mistral

        text = json.dumps({"flags": [], "low_confidence": []})
        lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": [
                                    {"type": "thinking", "thinking": "considering..."}
                                ]
                            }
                        }
                    ]
                }
            ),
            "",
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": [{"type": "text", "text": text}]}}]}
            ),
            "",
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                }
            ),
            "data: [DONE]",
        ]
        with patch(
            "ci_article_review.adapters.review.mistral.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = mistral.call("system", "user", "key")
        assert result["failed"] is False
        assert result["data"] == {"flags": [], "low_confidence": []}

    def test_successful_call(self):
        from ci_article_review.adapters.review import mistral

        content = {"flags": [], "low_confidence": []}
        with patch(
            "ci_article_review.adapters.review.mistral.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = mistral.call("system", "user", "key")
        assert result["failed"] is False
        assert "flags" in result["data"]
        assert "elapsed_seconds" in result

    def test_failed_call(self):
        from ci_article_review.adapters.review import mistral

        with patch(
            "ci_article_review.adapters.review.mistral.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("Timeout")
            result = mistral.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert "elapsed_seconds" in result


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------


class TestGemini:
    def _mock_response(self, content_dict):
        return _sse_mock(_sse_gemini_lines(content_dict))

    def test_successful_call(self):
        from ci_article_review.adapters.review import gemini

        content = {
            "confirmed": [],
            "outdated": [],
            "contradicted": [],
            "unverifiable": [],
            "primary_source_needed": [],
        }
        with patch(
            "ci_article_review.adapters.review.gemini.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = gemini.call("system", "user", "key")
        assert result["failed"] is False
        assert "confirmed" in result["data"]
        assert result["tokens"]["prompt"] == 120
        assert "elapsed_seconds" in result

    def test_thought_parts_excluded_from_stream(self):
        # Parts flagged thought:true are internal reasoning and must not pollute the JSON.
        from ci_article_review.adapters.review import gemini

        text = json.dumps({"confirmed": ["x"]})
        lines = [
            "data: "
            + json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "reasoning...", "thought": True}]
                            }
                        }
                    ]
                }
            ),
            "",
            "data: "
            + json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": text}]}}],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
                }
            ),
        ]
        with patch(
            "ci_article_review.adapters.review.gemini.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = gemini.call("system", "user", "key")
        assert result["failed"] is False
        assert result["data"] == {"confirmed": ["x"]}

    def test_no_candidates(self):
        from ci_article_review.adapters.review import gemini

        # An empty stream (no text parts) is the streaming equivalent of no candidates.
        empty = _sse_mock(
            [
                "data: "
                + json.dumps(
                    {"candidates": [{"content": {"parts": []}}], "usageMetadata": {}}
                )
            ]
        )
        with patch(
            "ci_article_review.adapters.review.gemini.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = empty
            result = gemini.call("system", "user", "key")
        assert result["failed"] is True

    def test_multiple_interleaved_thought_parts_excluded(self):
        # Thought and non-thought parts can arrive interleaved across several
        # chunks rather than as one contiguous thought block up front — the
        # accumulator must skip every thought-flagged part regardless of order.
        from ci_article_review.adapters.review import gemini

        text = json.dumps({"confirmed": ["x"], "outdated": []})
        half = len(text) // 2
        lines = []

        def _chunk(piece, thought=False, usage=None):
            cand = {"content": {"parts": [{"text": piece, "thought": thought}]}}
            if not thought:
                cand["content"]["parts"][0].pop("thought")
            obj = {"candidates": [cand]}
            if usage:
                obj["usageMetadata"] = usage
            return "data: " + json.dumps(obj)

        lines.append(_chunk("first reasoning step...", thought=True))
        lines.append("")
        lines.append(_chunk(text[:half]))
        lines.append("")
        lines.append(_chunk("second reasoning step...", thought=True))
        lines.append("")
        lines.append(
            _chunk(
                text[half:],
                usage={"promptTokenCount": 1, "candidatesTokenCount": 1},
            )
        )
        with patch(
            "ci_article_review.adapters.review.gemini.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = gemini.call("system", "user", "key")
        assert result["failed"] is False
        assert result["data"] == {"confirmed": ["x"], "outdated": []}

    def test_max_tokens_truncation_reported_distinctly(self):
        # A finishReason=MAX_TOKENS on genuinely truncated (unparseable) JSON
        # should be reported as a distinct, diagnosable failure instead of the
        # generic "Malformed JSON response" message.
        from ci_article_review.adapters.review import gemini

        truncated = '{"confirmed": ["x", "y", "z'  # cut off mid-string, invalid JSON
        lines = [
            "data: "
            + json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": truncated}]},
                            "finishReason": "MAX_TOKENS",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 50,
                    },
                }
            )
        ]
        with patch(
            "ci_article_review.adapters.review.gemini.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = gemini.call("system", "user", "key")
        assert result["failed"] is True
        assert "MAX_TOKENS" in result["error"]
        assert result["raw"] == truncated

    def test_api_key_redacted_in_errors(self):
        from ci_article_review.adapters.review.gemini import _redact_key

        api_key = "super-secret-key-abc123"
        error_with_key = f"HTTPError at https://example.com?key={api_key} returned 500"
        redacted = _redact_key(error_with_key, api_key)
        assert api_key not in redacted
        assert "[REDACTED]" in redacted

    def test_key_not_redacted_when_absent(self):
        from ci_article_review.adapters.review.gemini import _redact_key

        result = _redact_key("Some generic error message", "mykey")
        assert result == "Some generic error message"

    def test_timeout_error_message_is_domain_agnostic(self, caplog):
        # gemini.call() is invoked once per (model, domain) pair but is never told
        # which domain it's running (see pipeline.py's _run_domain — no domain arg
        # is passed to adapter.call()). The read-gap timeout log line must not name
        # a specific domain (e.g. "fact-check") since it would be wrong whenever
        # Gemini runs any other domain, such as completeness.
        from ci_article_review.adapters.review import gemini

        with patch(
            "ci_article_review.adapters.review.gemini.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = requests.exceptions.Timeout(
                "Read timed out."
            )
            with caplog.at_level("ERROR"):
                result = gemini.call("system", "user", "key")

        assert result["failed"] is True
        error_text = " ".join(r.message for r in caplog.records)
        assert "fact-check" not in error_text.lower(), (
            f"log message names a specific domain the adapter doesn't know: {error_text!r}"
        )
        assert "stream_read_timeout" in error_text, (
            f"log message should point at stream_read_timeout (the actual read-gap "
            f"control), not timeout_seconds: {error_text!r}"
        )

    def test_maximum_preset_gemini_has_stream_read_timeout_override(self):
        # The maximum preset stacks grounding (search) with thinking_budget: 16000
        # (extended reasoning) — two independent silent-period sources. A live
        # Vertex AI run timed out at 205.78s against the bare grounded default of
        # 160s. Guard against that config regressing back to the bare default.
        from ci_article_review.config_loader import _COST_PRESETS

        gemini_cfg = _COST_PRESETS["maximum"]["models"]["gemini"]
        assert gemini_cfg.get("thinking_budget") == 16000
        assert gemini_cfg.get("stream_read_timeout", 0) > 160, (
            "maximum preset's gemini entry combines grounding with thinking_budget "
            "and needs a stream_read_timeout override above the bare 160s grounded "
            "default — see the 205.78s live timeout this regresses against."
        )


# ---------------------------------------------------------------------------
# Grok adapter
# ---------------------------------------------------------------------------


class TestGrok:
    def _mock_response(self, content_dict):
        return _sse_mock(
            _sse_chat_lines(
                content_dict, usage={"prompt_tokens": 90, "completion_tokens": 45}
            )
        )

    def test_successful_call(self):
        from ci_article_review.adapters.review import grok

        content = {
            "most_vulnerable_claim": {
                "passage": "test",
                "attack_vector": "x",
                "supporting_evidence_for_attack": "y",
            },
            "highest_audience_risk": {
                "passage": "test",
                "risk": "z",
                "audience_segment": "all",
            },
            "highest_credibility_risk": {
                "passage": "test",
                "risk": "w",
                "attack_vector": "v",
            },
        }
        with patch(
            "ci_article_review.adapters.review.grok.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = grok.call("system", "user", "key")
        assert result["failed"] is False
        assert "most_vulnerable_claim" in result["data"]
        assert "elapsed_seconds" in result

    def test_failed_call(self):
        from ci_article_review.adapters.review import grok

        with patch(
            "ci_article_review.adapters.review.grok.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("Connection refused")
            result = grok.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert "elapsed_seconds" in result

    def test_model_override(self):
        from ci_article_review.adapters.review import grok

        content = {
            "most_vulnerable_claim": {},
            "highest_audience_risk": {},
            "highest_credibility_risk": {},
        }
        with patch(
            "ci_article_review.adapters.review.grok.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = grok.call("system", "user", "key", model="grok-2-latest")
        assert result["model"] == "grok-2-latest"


# ---------------------------------------------------------------------------
# Claude adapter
# ---------------------------------------------------------------------------


class TestClaude:
    def _mock_response(self, content_dict):
        return _sse_mock(
            _sse_anthropic_lines(content_dict, input_tokens=100, output_tokens=50)
        )

    def test_successful_call(self):
        from ci_article_review.adapters.review import claude

        content = {
            "flags": [
                {
                    "passage": "test",
                    "problem": "weak logic",
                    "suggested_rewrite": "stronger",
                }
            ],
            "low_confidence": [],
        }
        with patch(
            "ci_article_review.adapters.review.claude.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = claude.call("system", "user", "key")
        assert result["failed"] is False
        assert result["data"]["flags"][0]["passage"] == "test"
        assert result["tokens"]["prompt"] == 100
        assert result["tokens"]["completion"] == 50
        assert "elapsed_seconds" in result
        # stream must be requested in the Anthropic request body.
        assert mock_session.post.call_args.kwargs["json"]["stream"] is True

    def test_thinking_deltas_excluded(self):
        # Adaptive/extended reasoning streams thinking_delta events; only text_delta
        # forms the answer JSON.
        from ci_article_review.adapters.review import claude

        text = json.dumps({"flags": [], "low_confidence": []})
        lines = [
            "data: "
            + json.dumps(
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 100, "output_tokens": 0}},
                }
            ),
            "",
            "data: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "weighing..."},
                }
            ),
            "",
            "data: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": text},
                }
            ),
            "",
            "data: "
            + json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 42},
                }
            ),
        ]
        with patch(
            "ci_article_review.adapters.review.claude.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = claude.call("system", "user", "key")
        assert result["failed"] is False
        assert result["data"] == {"flags": [], "low_confidence": []}
        assert result["tokens"]["completion"] == 42  # from message_delta

    def test_failed_call(self):
        from ci_article_review.adapters.review import claude

        with patch(
            "ci_article_review.adapters.review.claude.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("Connection error")
            result = claude.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert "elapsed_seconds" in result

    def test_malformed_json(self):
        from ci_article_review.adapters.review import claude

        with patch(
            "ci_article_review.adapters.review.claude.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(
                _sse_anthropic_lines("not json at all")
            )
            result = claude.call("system", "user", "key")
        assert result["failed"] is True
        assert result["raw"] == "not json at all"

    def test_model_override(self):
        from ci_article_review.adapters.review import claude

        content = {"flags": [], "low_confidence": []}
        with patch(
            "ci_article_review.adapters.review.claude.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = claude.call("system", "user", "key", model="claude-sonnet-4-5")
        assert result["model"] == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# Perplexity JSON extraction (sonar-reasoning-pro <think> preambles / fences)
# ---------------------------------------------------------------------------


class TestPerplexityExtractJson:
    def test_plain_json(self):
        from ci_article_review.adapters.review.perplexity import _extract_json

        assert _extract_json('{"flags": []}') == {"flags": []}

    def test_code_fence(self):
        from ci_article_review.adapters.review.perplexity import _extract_json

        assert _extract_json('```json\n{"flags": [1]}\n```') == {"flags": [1]}

    def test_think_preamble(self):
        # The observed failure mode: a reasoning block before the JSON.
        from ci_article_review.adapters.review.perplexity import _extract_json

        raw = '<think>\nLet me reason about this claim...\nstep two\n</think>\n{"verdict": "confirmed"}'
        assert _extract_json(raw) == {"verdict": "confirmed"}

    def test_prose_before_and_after(self):
        from ci_article_review.adapters.review.perplexity import _extract_json

        raw = 'Here is my analysis:\n{"a": 1, "b": [2, 3]}\nHope that helps!'
        assert _extract_json(raw) == {"a": 1, "b": [2, 3]}

    def test_unrecoverable_returns_none(self):
        from ci_article_review.adapters.review.perplexity import _extract_json

        assert _extract_json("no json here at all") is None
        assert _extract_json("") is None

    def test_multiple_think_blocks(self):
        # sonar-reasoning-pro can emit more than one <think> block.
        from ci_article_review.adapters.review.perplexity import _extract_json

        raw = (
            "<think>first pass</think>\n"
            "<think>second pass, reconsidering</think>\n"
            '{"verdict": "confirmed"}'
        )
        assert _extract_json(raw) == {"verdict": "confirmed"}

    def test_think_block_containing_braces_does_not_corrupt_span_match(self):
        # A reasoning block that itself mentions braces (e.g. discussing the
        # target JSON schema) must not widen the outermost-{...} span past the
        # real payload.
        from ci_article_review.adapters.review.perplexity import _extract_json

        raw = (
            '<think>The schema should look like {"verdict": ...} '
            "so I need to produce that.</think>\n"
            '{"verdict": "confirmed"}'
        )
        assert _extract_json(raw) == {"verdict": "confirmed"}

    def test_worst_case_think_fence_and_leading_prose(self):
        # The documented worst case: leading prose, a <think> block, AND a
        # markdown fence, all in the same response.
        from ci_article_review.adapters.review.perplexity import _extract_json

        raw = (
            "Sure, here is my analysis.\n"
            "<think>\nLet me think about the claim: {this is not json}\n</think>\n"
            "```json\n"
            '{"verdict": "confirmed", "citations": [1, 2]}\n'
            "```\n"
            "Let me know if you need anything else!"
        )
        assert _extract_json(raw) == {
            "verdict": "confirmed",
            "citations": [1, 2],
        }


class TestPerplexityStreamFailures:
    """Full call() coverage for failure shapes upstream of JSON parsing.

    Perplexity's observed live failure had zero captured token usage — unlike
    Gemini's, which had full usage. That points to a failure earlier in the
    response pipeline than "got text, couldn't parse it as JSON". These tests
    cover the two upstream shapes: an SSE stream that produces no usable
    content at all, and an in-band {"error": ...} event instead of choices.
    """

    def test_empty_stream_reports_distinct_error_not_malformed_json(self):
        from ci_article_review.adapters.review import perplexity

        # A stream that produces no choices/content and no usage at all —
        # e.g. a dropped connection after headers but before any data.
        lines = ["data: " + json.dumps({}), "data: [DONE]"]
        with patch(
            "ci_article_review.adapters.review.perplexity.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = perplexity.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert result["error"] != "Malformed JSON response"
        assert "empty" in result["error"].lower()
        assert result["tokens"] == {}

    def test_inband_stream_error_event_reported_distinctly(self):
        from ci_article_review.adapters.review import perplexity

        lines = [
            "data: "
            + json.dumps({"error": {"message": "rate limit exceeded", "code": 429}}),
            "data: [DONE]",
        ]
        with patch(
            "ci_article_review.adapters.review.perplexity.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = perplexity.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert "rate limit exceeded" in result["error"]
        assert result["error"] != "Malformed JSON response"

    def test_inband_400_invalid_request_logs_payload_diagnostics(self, caplog):
        # Live occurrence (2026-08-06): perplexity:voice_style at the maximum
        # preset got a 400 invalid_request in-band SSE error with no field-level
        # detail. On that shape specifically, the adapter should log a redacted
        # excerpt of the outgoing payload so a recurrence is diagnosable without
        # another investigation.
        from ci_article_review.adapters.review import perplexity

        lines = [
            "data: "
            + json.dumps(
                {
                    "error": {
                        "message": "invalid request",
                        "type": "invalid_request",
                        "code": 400,
                    }
                }
            ),
            "data: [DONE]",
        ]
        with patch(
            "ci_article_review.adapters.review.perplexity.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            with caplog.at_level("ERROR"):
                result = perplexity.call(
                    "voice/style system prompt",
                    "user draft content",
                    "sk-secret-key",
                    retry=False,
                )
        assert result["failed"] is True
        assert "invalid request" in result["error"]
        diagnostic_logs = "\n".join(
            r.message for r in caplog.records if "invalid_request payload" in r.message
        )
        assert "voice/style system prompt" in diagnostic_logs
        assert "user draft content" in diagnostic_logs
        assert "sk-secret-key" not in diagnostic_logs

    def test_payload_has_no_extraneous_keys_for_voice_style_config(self):
        # Guards against pipeline-internal config keys (timeout_seconds, prompts,
        # enabled, etc.) leaking into the actual request body — the request must
        # only ever carry parameters Perplexity's API accepts.
        from ci_article_review.adapters.review import perplexity

        captured = {}

        def _capture_post(url, headers, json, stream, timeout):
            captured["payload"] = json
            lines = _sse_chat_lines('{"flags": []}')
            return _sse_mock(lines)

        with patch(
            "ci_article_review.adapters.review.perplexity.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = _capture_post
            perplexity.call(
                "system prompt for voice_style" * 50,
                "user draft" * 500,
                "key",
                retry=False,
                provider_config={
                    "model": "sonar-reasoning-pro",
                    "prompts": ["voice_style"],
                    "timeout_seconds": 375,
                    "enabled": True,
                    "reasoning_effort": "medium",
                },
            )

        payload = captured["payload"]
        assert set(payload.keys()) <= {
            "model",
            "messages",
            "temperature",
            "stream",
            "stream_options",
            "reasoning_effort",
        }
        assert payload["model"] == "sonar-reasoning-pro"
        assert payload["reasoning_effort"] == "medium"
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]
        assert payload["messages"][0]["content"].startswith(
            "system prompt for voice_style"
        )
        assert "response_format" not in payload

    def test_genuine_malformed_json_still_reported_as_such(self):
        # Content was received (and usage captured) but doesn't parse as JSON
        # even after the extraction fallbacks -- this remains "Malformed JSON
        # response" and still carries the raw text for diagnostics.
        from ci_article_review.adapters.review import perplexity

        lines = _sse_chat_lines(
            "This is prose with no JSON payload anywhere in it at all.",
            usage={"prompt_tokens": 50, "completion_tokens": 20},
        )
        with patch(
            "ci_article_review.adapters.review.perplexity.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(lines)
            result = perplexity.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert result["error"] == "Malformed JSON response"
        assert (
            result["raw"] == "This is prose with no JSON payload anywhere in it at all."
        )
        assert result["tokens"] == {"prompt_tokens": 50, "completion_tokens": 20}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_invalid_publication_name_raises(self):
        from ci_article_review.config_loader import validate_publication_name

        with pytest.raises(ValueError, match="Invalid publication name"):
            validate_publication_name("../../etc/shadow")

    # --- model config normalization ---

    def test_normalize_simple_string_gemini(self):
        from ci_article_review.config_loader import _normalize_model_configs

        result = _normalize_model_configs({"gemini": "gemini-2.5-flash"})
        assert result["gemini"]["provider"] == "ai_studio"
        assert result["gemini"]["model"] == "gemini-2.5-flash"

    def test_normalize_simple_string_openai(self):
        from ci_article_review.config_loader import _normalize_model_configs

        result = _normalize_model_configs({"openai": "gpt-4o"})
        assert result["openai"]["provider"] == "openai"
        assert result["openai"]["model"] == "gpt-4o"

    def test_normalize_extended_dict_passthrough(self):
        from ci_article_review.config_loader import _normalize_model_configs

        cfg = {
            "gemini": {
                "provider": "vertex_ai",
                "model": "gemini-2.5-flash",
                "project": "my-proj",
            }
        }
        result = _normalize_model_configs(cfg)
        assert result["gemini"]["provider"] == "vertex_ai"
        assert result["gemini"]["project"] == "my-proj"

    def test_normalize_dict_without_provider_gets_default(self):
        from ci_article_review.config_loader import _normalize_model_configs

        result = _normalize_model_configs(
            {"mistral": {"model": "mistral-large-latest"}}
        )
        assert result["mistral"]["provider"] == "mistral"

    def test_normalize_mixed_forms(self):
        from ci_article_review.config_loader import _normalize_model_configs

        raw = {
            "openai": "gpt-4o",
            "gemini": {
                "provider": "vertex_ai",
                "model": "gemini-2.5-flash",
                "project": "p",
            },
        }
        result = _normalize_model_configs(raw)
        assert result["openai"]["provider"] == "openai"
        assert result["gemini"]["provider"] == "vertex_ai"

    def test_normalize_simple_string_claude(self):
        from ci_article_review.config_loader import _normalize_model_configs

        result = _normalize_model_configs({"claude": "claude-opus-4-5"})
        assert result["claude"]["provider"] == "anthropic"
        assert result["claude"]["model"] == "claude-opus-4-5"

    def test_normalize_preserves_enabled_flag(self):
        from ci_article_review.config_loader import _normalize_model_configs

        result = _normalize_model_configs(
            {"grok": {"model": "grok-3-latest", "enabled": False}}
        )
        assert result["grok"]["enabled"] is False
        assert result["grok"]["provider"] == "grok"

    def test_normalize_enabled_defaults_absent_for_simple_form(self):
        from ci_article_review.config_loader import _normalize_model_configs

        result = _normalize_model_configs({"openai": "gpt-4o"})
        # Simple form has no enabled key — caller should default to True
        assert "enabled" not in result["openai"]

    def test_normalize_empty_input(self):
        from ci_article_review.config_loader import _normalize_model_configs

        assert _normalize_model_configs({}) == {}
        assert _normalize_model_configs(None) == {}

    def test_invalid_publication_name_with_slash(self):
        from ci_article_review.config_loader import validate_publication_name

        with pytest.raises(ValueError):
            validate_publication_name("my/blog")

    def test_valid_publication_name(self):
        from ci_article_review.config_loader import validate_publication_name

        validate_publication_name("my-blog")
        validate_publication_name("myblog")
        validate_publication_name("my_blog_2024")

    def test_env_var_missing_gives_helpful_error(self):
        from ci_article_review.config_loader import _resolve_env

        env_key = "PIPELINE_TEST_MISSING_VAR_XYZ"
        if env_key in os.environ:
            del os.environ[env_key]
        with pytest.raises(ValueError, match="not set"):
            _resolve_env(f"${{{env_key}}}")


# ---------------------------------------------------------------------------
# Provider dispatch — verify correct backend is selected
# ---------------------------------------------------------------------------


class TestProviderDispatch:
    """Verify that provider_config routes each adapter to the right backend."""

    # --- Gemini ---

    def test_gemini_default_uses_aistudio(self):
        from ci_article_review.adapters.review import gemini

        with (
            patch("ci_article_review.adapters.review.gemini._call_aistudio") as mock_ai,
            patch("ci_article_review.adapters.review.gemini._call_vertex") as mock_vx,
        ):
            mock_ai.return_value = {
                "failed": False,
                "data": {},
                "model": "x",
                "tokens": {},
                "elapsed_seconds": 0,
            }
            gemini.call("sys", "usr", "key", retry=False)
        mock_ai.assert_called_once()
        mock_vx.assert_not_called()

    def test_gemini_vertex_ai_dispatches_to_vertex(self):
        from ci_article_review.adapters.review import gemini

        cfg = {
            "provider": "vertex_ai",
            "model": "gemini-2.5-flash",
            "project": "p",
            "location": "us-central1",
        }
        with (
            patch("ci_article_review.adapters.review.gemini._call_vertex") as mock_vx,
            patch("ci_article_review.adapters.review.gemini._call_aistudio") as mock_ai,
        ):
            mock_vx.return_value = {
                "failed": False,
                "data": {},
                "model": "x",
                "tokens": {},
                "elapsed_seconds": 0,
            }
            gemini.call("sys", "usr", "key", retry=False, provider_config=cfg)
        mock_vx.assert_called_once()
        mock_ai.assert_not_called()

    def test_gemini_model_from_provider_config(self):
        """Model name in provider_config is used when no explicit model arg is passed."""
        from ci_article_review.adapters.review import gemini

        cfg = {"provider": "ai_studio", "model": "gemini-2.5-pro"}
        captured = {}

        def fake_aistudio(sys, usr, key, model, **kw):
            captured["model"] = model
            return {
                "failed": False,
                "data": {},
                "model": model,
                "tokens": {},
                "elapsed_seconds": 0,
            }

        with patch(
            "ci_article_review.adapters.review.gemini._call_aistudio",
            side_effect=fake_aistudio,
        ):
            gemini.call("sys", "usr", "key", retry=False, provider_config=cfg)
        assert captured["model"] == "gemini-2.5-pro"

    # --- OpenAI ---

    def test_openai_default_uses_openai_backend(self):
        from ci_article_review.adapters.review import openai as oai

        with (
            patch("ci_article_review.adapters.review.openai._call_openai") as mock_oa,
            patch("ci_article_review.adapters.review.openai._call_azure") as mock_az,
        ):
            mock_oa.return_value = {
                "failed": False,
                "data": {},
                "model": "x",
                "tokens": {},
                "elapsed_seconds": 0,
            }
            oai.call("sys", "usr", "key", retry=False)
        mock_oa.assert_called_once()
        mock_az.assert_not_called()

    def test_openai_azure_dispatches_to_azure(self):
        from ci_article_review.adapters.review import openai as oai

        cfg = {
            "provider": "azure",
            "model": "gpt-4o",
            "endpoint": "https://res.openai.azure.com",
            "deployment": "my-dep",
        }
        with (
            patch("ci_article_review.adapters.review.openai._call_azure") as mock_az,
            patch("ci_article_review.adapters.review.openai._call_openai") as mock_oa,
        ):
            mock_az.return_value = {
                "failed": False,
                "data": {},
                "model": "x",
                "tokens": {},
                "elapsed_seconds": 0,
            }
            oai.call("sys", "usr", "key", retry=False, provider_config=cfg)
        mock_az.assert_called_once()
        mock_oa.assert_not_called()

    def test_openai_azure_missing_endpoint_returns_failure(self):
        from ci_article_review.adapters.review import openai as oai

        cfg = {"provider": "azure", "model": "gpt-4o"}  # no endpoint or deployment
        result = oai.call("sys", "usr", "key", retry=False, provider_config=cfg)
        assert result["failed"] is True
        assert (
            "endpoint" in result["error"].lower()
            or "deployment" in result["error"].lower()
        )

    # --- Mistral ---

    def test_mistral_default_uses_laplateforme(self):
        from ci_article_review.adapters.review import mistral

        with (
            patch(
                "ci_article_review.adapters.review.mistral._call_laplateforme"
            ) as mock_lp,
            patch("ci_article_review.adapters.review.mistral._call_azure") as mock_az,
        ):
            mock_lp.return_value = {
                "failed": False,
                "data": {},
                "model": "x",
                "tokens": {},
                "elapsed_seconds": 0,
            }
            mistral.call("sys", "usr", "key", retry=False)
        mock_lp.assert_called_once()
        mock_az.assert_not_called()

    def test_mistral_azure_dispatches_to_azure(self):
        from ci_article_review.adapters.review import mistral

        cfg = {
            "provider": "azure",
            "model": "mistral-large-latest",
            "endpoint": "https://Mistral-Large-abc.eastus2.inference.ai.azure.com",
        }
        with (
            patch("ci_article_review.adapters.review.mistral._call_azure") as mock_az,
            patch(
                "ci_article_review.adapters.review.mistral._call_laplateforme"
            ) as mock_lp,
        ):
            mock_az.return_value = {
                "failed": False,
                "data": {},
                "model": "x",
                "tokens": {},
                "elapsed_seconds": 0,
            }
            mistral.call("sys", "usr", "key", retry=False, provider_config=cfg)
        mock_az.assert_called_once()
        mock_lp.assert_not_called()

    def test_mistral_azure_missing_endpoint_returns_failure(self):
        from ci_article_review.adapters.review import mistral

        cfg = {"provider": "azure", "model": "mistral-large-latest"}  # no endpoint
        result = mistral.call("sys", "usr", "key", retry=False, provider_config=cfg)
        assert result["failed"] is True
        assert "endpoint" in result["error"].lower()

    # --- Grok (no enterprise tier yet — just verify provider_config accepted) ---

    def test_grok_accepts_provider_config_without_error(self):
        from ci_article_review.adapters.review import grok

        content = {
            "most_vulnerable_claim": {},
            "highest_audience_risk": {},
            "highest_credibility_risk": {},
        }
        cfg = {"provider": "grok", "model": "grok-3-latest"}
        with patch(
            "ci_article_review.adapters.review.grok.requests.Session"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = _sse_mock(
                _sse_chat_lines(
                    content, usage={"prompt_tokens": 10, "completion_tokens": 5}
                )
            )
            result = grok.call("sys", "usr", "key", retry=False, provider_config=cfg)
        assert result["failed"] is False


# ---------------------------------------------------------------------------
# History / slug safety
# ---------------------------------------------------------------------------


class TestHistory:
    def test_slug_strips_special_chars(self):
        from ci_article_review.history import _slug

        assert "/" not in _slug("My Article: Part 1/2")
        assert ":" not in _slug("My Article: Part 1/2")

    def test_slug_windows_reserved_names(self):
        from ci_article_review.history import _slug

        for reserved in ("CON", "con", "PRN", "AUX", "NUL", "COM1", "LPT9"):
            result = _slug(reserved)
            assert result.lower() not in {
                "con",
                "prn",
                "aux",
                "nul",
                "com1",
                "com2",
                "com3",
                "com4",
                "com5",
                "com6",
                "com7",
                "com8",
                "com9",
                "lpt1",
                "lpt2",
                "lpt3",
                "lpt4",
                "lpt5",
                "lpt6",
                "lpt7",
                "lpt8",
                "lpt9",
            }, f"Reserved name {reserved!r} was not escaped — got {result!r}"

    def test_slug_path_traversal_neutralized(self):
        from ci_article_review.history import _slug

        result = _slug("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result


# ---------------------------------------------------------------------------
# History — write error handling
# ---------------------------------------------------------------------------


class TestHistoryWriteErrors:
    def test_report_write_failure_returns_none_paths(self, tmp_path):
        import ci_article_review.history as hist

        report = {"article_title": "Test Article", "run_number": 1}
        with patch("builtins.open", side_effect=OSError("disk full")):
            paths = hist.save_run(str(tmp_path), "Test Article", 1, report, [])
        assert paths["report_path"] is None
        assert paths["corrections_path"] is None

    def test_corrections_write_failure_returns_report_path(self, tmp_path):
        import ci_article_review.history as hist

        report = {"article_title": "Test Article", "run_number": 1}
        call_count = {"n": 0}
        real_open = open

        def selective_fail(path, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 3:  # third open is the corrections log
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_fail):
            paths = hist.save_run(str(tmp_path), "Test Article", 1, report, [])
        assert paths["report_path"] is not None
        assert paths["corrections_path"] is None

    def test_markdown_write_failure_returns_other_paths(self, tmp_path):
        import ci_article_review.history as hist

        report = {"article_title": "Test Article", "run_number": 1}
        call_count = {"n": 0}
        real_open = open

        def selective_fail(path, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:  # second open is the markdown review
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_fail):
            paths = hist.save_run(str(tmp_path), "Test Article", 1, report, [])
        assert paths["report_path"] is not None
        assert paths["corrections_path"] is not None
        assert paths["markdown_path"] is None


# ---------------------------------------------------------------------------
# WordPress — term ID resolution
# ---------------------------------------------------------------------------


class TestWordPressTermResolution:
    def _mock_term_response(self, term_id, slug):
        mock = MagicMock()
        mock.ok = True
        mock.json.return_value = [{"id": term_id, "slug": slug}]
        mock.raise_for_status = MagicMock()
        return mock

    def _mock_empty_response(self):
        mock = MagicMock()
        mock.ok = True
        mock.json.return_value = []
        mock.raise_for_status = MagicMock()
        return mock

    def test_integer_ids_pass_through(self):
        from ci_article_review.adapters.cms.wordpress import _lookup_term_ids

        result = _lookup_term_ids(
            "http://site.com/wp-json/wp/v2", {}, "categories", [5, 12]
        )
        assert result == [5, 12]

    def test_slug_resolved_to_id(self):
        from ci_article_review.adapters.cms.wordpress import _lookup_term_ids

        with patch("ci_article_review.adapters.cms.wordpress.requests.get") as mock_get:
            mock_get.return_value = self._mock_term_response(42, "data-centers")
            result = _lookup_term_ids(
                "http://site.com/wp-json/wp/v2", {}, "categories", ["data-centers"]
            )
        assert result == [42]

    def test_unknown_slug_omitted_with_warning(self):
        from ci_article_review.adapters.cms.wordpress import _lookup_term_ids

        with patch("ci_article_review.adapters.cms.wordpress.requests.get") as mock_get:
            mock_get.return_value = self._mock_empty_response()
            result = _lookup_term_ids(
                "http://site.com/wp-json/wp/v2", {}, "tags", ["nonexistent-tag"]
            )
        assert result == []

    def test_mixed_ids_and_slugs(self):
        from ci_article_review.adapters.cms.wordpress import _lookup_term_ids

        with patch("ci_article_review.adapters.cms.wordpress.requests.get") as mock_get:
            mock_get.return_value = self._mock_term_response(99, "energy")
            result = _lookup_term_ids(
                "http://site.com/wp-json/wp/v2", {}, "tags", [7, "energy"]
            )
        assert 7 in result
        assert 99 in result

    def test_single_string_accepted(self):
        from ci_article_review.adapters.cms.wordpress import _lookup_term_ids

        with patch("ci_article_review.adapters.cms.wordpress.requests.get") as mock_get:
            mock_get.return_value = self._mock_term_response(3, "tech")
            result = _lookup_term_ids(
                "http://site.com/wp-json/wp/v2", {}, "categories", "tech"
            )
        assert result == [3]

    def test_empty_input_returns_empty(self):
        from ci_article_review.adapters.cms.wordpress import _lookup_term_ids

        assert (
            _lookup_term_ids("http://site.com/wp-json/wp/v2", {}, "categories", [])
            == []
        )
        assert (
            _lookup_term_ids("http://site.com/wp-json/wp/v2", {}, "categories", None)
            == []
        )


# ---------------------------------------------------------------------------
# Handoff parser — empty next_headers fix
# ---------------------------------------------------------------------------


class TestHandoffParser:
    def test_last_section_extracted(self):
        from ci_article_review.handoff_parser import _extract_section

        doc = "HEADER ONE\nfirst content\n\nHEADER TWO\nlast content here"
        result = _extract_section(doc, "HEADER TWO", next_headers=None)
        assert result == "last content here"

    def test_middle_section_extracted(self):
        from ci_article_review.handoff_parser import _extract_section

        doc = "HEADER ONE\nfirst content\n\nHEADER TWO\nmiddle content\n\nHEADER THREE\nthird content"
        result = _extract_section(doc, "HEADER TWO", next_headers=["HEADER THREE"])
        assert "middle content" in result
        assert "third content" not in result

    def test_missing_required_field_logs_warning(self):
        """Missing title/primary_claim/draft should log a warning, not crash."""
        from ci_article_review.handoff_parser import parse_draft_submission

        # Minimal doc with no Article: field and no DRAFT section
        doc = "DRAFT SUBMISSION HANDOFF\n\nPRIMARY CLAIM\nSome claim\n\nDRAFT\nArticle text here."
        with patch("ci_article_review.handoff_parser.log") as mock_log:
            result = parse_draft_submission(doc)
        # title is empty — should have warned
        warning_messages = [str(call) for call in mock_log.warning.call_args_list]
        assert any("Article:" in m for m in warning_messages)
        # draft was present — should not have warned about it
        assert result["draft"] == "Article text here."

    def test_missing_draft_logs_warning(self):
        from ci_article_review.handoff_parser import parse_draft_submission

        doc = "Article: My Article\n\nPRIMARY CLAIM\nSome claim here\n\n"
        with patch("ci_article_review.handoff_parser.log") as mock_log:
            result = parse_draft_submission(doc)
        warning_messages = [str(call) for call in mock_log.warning.call_args_list]
        assert any("DRAFT" in m for m in warning_messages)
        assert result["draft"] == ""


# ---------------------------------------------------------------------------
# check.py — provider dispatch
# ---------------------------------------------------------------------------


class TestCheckProviderDispatch:
    def _ok_response(self, body):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = body
        mock.raise_for_status = MagicMock()
        return mock

    def test_check_openai_calls_correct_url(self):
        import ci_article_review.check as check

        resp = self._ok_response({"choices": [{"message": {"content": "ok"}}]})
        with patch(
            "ci_article_review.check.requests.post", return_value=resp
        ) as mock_post:
            check.check_openai("key", "gpt-4o")
        url = mock_post.call_args[0][0]
        assert "api.openai.com" in url
        assert "gpt-4o" in str(mock_post.call_args)

    def test_check_openai_azure_uses_api_key_header(self):
        import ci_article_review.check as check

        resp = self._ok_response({"choices": [{"message": {"content": "ok"}}]})
        cfg = {
            "endpoint": "https://res.openai.azure.com",
            "deployment": "my-dep",
            "model": "gpt-4o",
        }
        with patch(
            "ci_article_review.check.requests.post", return_value=resp
        ) as mock_post:
            check.check_openai_azure("key", cfg)
        headers = mock_post.call_args[1]["headers"]
        assert "api-key" in headers
        assert "Authorization" not in headers
        url = mock_post.call_args[0][0]
        assert "my-dep" in url

    def test_check_openai_azure_missing_endpoint_raises(self):
        import ci_article_review.check as check

        with pytest.raises(Exception, match="endpoint"):
            check.check_openai_azure("key", {"deployment": "dep"})

    def test_check_mistral_azure_uses_bearer_auth(self):
        import ci_article_review.check as check

        resp = self._ok_response({"choices": [{"message": {"content": "ok"}}]})
        cfg = {
            "endpoint": "https://Mistral-abc.eastus2.inference.ai.azure.com",
            "model": "mistral-large",
        }
        with patch(
            "ci_article_review.check.requests.post", return_value=resp
        ) as mock_post:
            check.check_mistral_azure("key", cfg)
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer key"

    def test_check_claude_uses_x_api_key_header(self):
        import ci_article_review.check as check

        resp = self._ok_response({"content": [{"type": "text", "text": "ok"}]})
        with patch(
            "ci_article_review.check.requests.post", return_value=resp
        ) as mock_post:
            result = check.check_claude("key", "claude-opus-4-5")
        headers = mock_post.call_args[1]["headers"]
        assert headers["x-api-key"] == "key"
        assert "replied" in result


# ---------------------------------------------------------------------------
# Cross-adapter: the streaming read-gap timeout reaches the HTTP transport
# ---------------------------------------------------------------------------
#
# Under streaming the socket timeout changed meaning: it is now the (connect,
# read-gap) tuple where read-gap is the MAX STALL BETWEEN TOKENS, a small constant
# — NOT the sliding-scale timeout_seconds (which is now only the pipeline's
# wall-clock backstop). This guards the whole class of adapters at once:
#   1. every adapter passes stream=True and a (connect, read) tuple to .post();
#   2. the read-gap is the per-adapter constant, NOT inflated by timeout_seconds;
#   3. a per-model stream_read_timeout override is honored.
# Every review adapter shares the requests.Session().post(stream=, timeout=) shape.

# (module path, a representative model id). Default ai_studio path for gemini
# avoids the google.auth flow (vertex), keeping the harness uniform.
_REVIEW_ADAPTERS = [
    ("ci_article_review.adapters.review.openai", "gpt-5.5"),
    ("ci_article_review.adapters.review.grok", "grok-4.20-0309-reasoning"),
    ("ci_article_review.adapters.review.mistral", "mistral-medium-3-5"),
    ("ci_article_review.adapters.review.perplexity", "sonar-reasoning-pro"),
    ("ci_article_review.adapters.review.claude", "claude-opus-4-8"),
    ("ci_article_review.adapters.review.gemini", "gemini-2.5-flash"),
]

# Per-adapter default inter-token read-gap (seconds) when stream_read_timeout is
# NOT set. Grounded adapters (Gemini, Perplexity) default higher because the live
# web search runs before the first token arrives.
_DEFAULT_READ_GAPS = {
    "ci_article_review.adapters.review.openai": 120,
    "ci_article_review.adapters.review.grok": 120,
    "ci_article_review.adapters.review.mistral": 120,
    "ci_article_review.adapters.review.perplexity": 160,
    "ci_article_review.adapters.review.claude": 120,
    "ci_article_review.adapters.review.gemini": 160,
}


def _universal_sse_response():
    """A streamed response mock valid for every adapter's accumulator, so exactly
    one POST happens and no fallback/retry path is triggered. Carries an empty-JSON
    answer plus a usage chunk in each provider's SSE shape."""
    lines = [
        # OpenAI / Grok / Mistral / Perplexity (chat completions delta)
        "data: " + json.dumps({"choices": [{"delta": {"content": "{}"}}]}),
        "data: "
        + json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ),
        # Anthropic messages events
        "data: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 1, "output_tokens": 0}},
            }
        ),
        "data: "
        + json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "{}"},
            }
        ),
        "data: "
        + json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            }
        ),
        # Gemini streamGenerateContent chunk
        "data: "
        + json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            }
        ),
        "data: [DONE]",
    ]
    return _sse_mock(lines)


class TestStreamingTimeoutAcrossAdapters:
    @pytest.mark.parametrize("module_name,model", _REVIEW_ADAPTERS)
    def test_streams_with_read_gap_not_inflated_by_timeout_seconds(
        self, module_name, model
    ):
        import importlib

        mod = importlib.import_module(module_name)
        sess = MagicMock()
        sess.post.return_value = _universal_sse_response()
        with patch(f"{module_name}.requests.Session", return_value=sess):
            # A huge timeout_seconds must NOT become the socket read timeout.
            mod.call(
                "system",
                "user",
                "key",
                provider_config={"model": model, "timeout_seconds": 819},
            )
        assert sess.post.called, f"{module_name}: .post() was never called"
        kwargs = sess.post.call_args.kwargs
        assert kwargs.get("stream") is True, f"{module_name} did not pass stream=True"
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, tuple) and len(timeout) == 2, (
            f"{module_name} must pass a (connect, read) tuple under streaming — got {timeout!r}"
        )
        read_gap = timeout[1]
        assert read_gap == _DEFAULT_READ_GAPS[module_name], (
            f"{module_name} read-gap should be the constant "
            f"{_DEFAULT_READ_GAPS[module_name]}s regardless of timeout_seconds=819 — got {read_gap!r}. "
            f"timeout_seconds must not inflate the inter-token socket timeout."
        )

    @pytest.mark.parametrize("module_name,model", _REVIEW_ADAPTERS)
    def test_default_read_gap_when_unset(self, module_name, model):
        import importlib

        mod = importlib.import_module(module_name)
        sess = MagicMock()
        sess.post.return_value = _universal_sse_response()
        with patch(f"{module_name}.requests.Session", return_value=sess):
            mod.call("system", "user", "key", provider_config={"model": model})
        timeout = sess.post.call_args.kwargs.get("timeout")
        assert timeout[1] == _DEFAULT_READ_GAPS[module_name], (
            f"{module_name} default read-gap changed — got {timeout[1]!r}, "
            f"expected {_DEFAULT_READ_GAPS[module_name]}"
        )

    @pytest.mark.parametrize("module_name,model", _REVIEW_ADAPTERS)
    def test_stream_read_timeout_override_honored(self, module_name, model):
        import importlib

        mod = importlib.import_module(module_name)
        sess = MagicMock()
        sess.post.return_value = _universal_sse_response()
        with patch(f"{module_name}.requests.Session", return_value=sess):
            mod.call(
                "system",
                "user",
                "key",
                provider_config={"model": model, "stream_read_timeout": 222},
            )
        timeout = sess.post.call_args.kwargs.get("timeout")
        assert timeout[1] == 222, (
            f"{module_name} ignored stream_read_timeout override — got {timeout[1]!r} (expected 222)"
        )
