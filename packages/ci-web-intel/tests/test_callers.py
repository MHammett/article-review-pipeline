"""Tests for callers.py — multi-model routing, SSE accumulation, error handling."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import json as _json


def _mock_sse_response(lines: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = [line.encode() for line in lines]
    return resp


def _anthropic_sse(text: str) -> list[str]:
    t = _json.dumps(text)  # properly escaped JSON string value including surrounding quotes
    return [
        'data: {"type": "message_start", "message": {"usage": {"input_tokens": 100, "output_tokens": 0}}}',
        f'data: {{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": {t}}}}}',
        'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 20}}',
        "data: [DONE]",
    ]


def _openai_sse(text: str) -> list[str]:
    t = _json.dumps(text)
    return [
        f'data: {{"choices": [{{"delta": {{"content": {t}}}, "finish_reason": null}}]}}',
        'data: {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}',
        "data: [DONE]",
    ]


def _gemini_sse(text: str) -> list[str]:
    t = _json.dumps(text)
    return [
        f'data: {{"candidates": [{{"content": {{"parts": [{{"text": {t}}}]}}}}], "usageMetadata": {{"promptTokenCount": 100, "candidatesTokenCount": 20}}}}',
        "data: [DONE]",
    ]


_MOCK_USER_CONFIG = {
    "models": {
        "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "openai": {"provider": "openai", "model": "gpt-5.4"},
        "gemini": {"provider": "ai_studio", "model": "gemini-2.5-flash"},
    },
    "api_keys": {
        "claude": {"api_key": "test-key-claude"},
        "openai": {"api_key": "test-key-openai"},
        "gemini": {"api_key": "test-key-gemini"},
    },
}


class TestCallOneAnthropic:
    def test_anthropic_success(self):
        """call_one Anthropic: mock SSE → content and tokens returned."""
        from callers import call_one, clear_api_call_log
        clear_api_call_log()

        resp = _mock_sse_response(_anthropic_sse('{"voice_profile": "test"}'))
        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value.__enter__ = lambda s: mock_session
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = resp

            result = call_one(
                "claude",
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                {"claude": {"api_key": "sk-test"}},
                "system",
                "user",
            )

        assert not result.get("failed"), result.get("error")
        assert result["tokens"]["prompt"] == 100
        assert result["tokens"]["completion"] == 20
        assert '{"voice_profile": "test"}' in result["content"]

    def test_anthropic_http_error(self):
        """call_one returns failed=True on HTTP error; does not raise."""
        from callers import call_one, clear_api_call_log
        clear_api_call_log()

        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("Connection refused")

            result = call_one(
                "claude",
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                {"claude": {"api_key": "sk-test"}},
                "system",
                "user",
            )

        assert result["failed"] is True
        assert "error" in result


class TestCallOneOpenAI:
    def test_openai_success(self):
        """call_one OpenAI: mock chat completions SSE → correct content."""
        from callers import call_one, clear_api_call_log
        clear_api_call_log()

        resp = _mock_sse_response(_openai_sse('{"voice_profile": "openai result"}'))
        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = resp

            result = call_one(
                "openai",
                {"provider": "openai", "model": "gpt-5.4"},
                {"openai": {"api_key": "sk-test"}},
                "system",
                "user",
            )

        assert not result.get("failed")
        assert "openai result" in result.get("content", "")

    def test_openai_server_error_returns_failed(self):
        """call_one returns failed=True on 5xx; does not raise."""
        from callers import call_one, clear_api_call_log
        clear_api_call_log()

        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("503 Server Error")

            result = call_one(
                "openai",
                {"provider": "openai", "model": "gpt-5.4"},
                {"openai": {"api_key": "sk-test"}},
                "system",
                "user",
            )

        assert result["failed"] is True


class TestCallOneGemini:
    def test_gemini_success(self):
        """call_one Gemini: mock SSE → usageMetadata mapped to tokens."""
        from callers import call_one, clear_api_call_log
        clear_api_call_log()

        resp = _mock_sse_response(_gemini_sse('{"voice_profile": "gemini result"}'))
        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = resp

            result = call_one(
                "gemini",
                {"provider": "ai_studio", "model": "gemini-2.5-flash"},
                {"gemini": {"api_key": "AIza-test"}},
                "system",
                "user",
            )

        assert not result.get("failed")
        assert result["tokens"]["prompt"] == 100
        assert result["tokens"]["completion"] == 20


class TestCallAll:
    def test_call_all_three_models(self):
        """call_all with 3 models: all 3 called in parallel; results keyed by model name."""
        from callers import call_all, clear_api_call_log
        clear_api_call_log()

        def _fake_call_one(model_name, model_cfg, api_keys, system, user, pass_name=""):
            return {"content": f"{model_name}_result", "failed": False, "tokens": {}, "elapsed": 0.1, "model": model_name}

        with patch("callers.call_one", side_effect=_fake_call_one):
            results = call_all(
                system_prompt="system",
                user_prompt="user",
                user_config=_MOCK_USER_CONFIG,
                pass_name="test",
            )

        assert "claude" in results
        assert "openai" in results
        assert "gemini" in results
        assert results["claude"]["content"] == "claude_result"

    def test_call_all_subset(self):
        """call_all with models=["claude"]: only Claude called."""
        from callers import call_all, clear_api_call_log
        clear_api_call_log()

        called = []

        def _fake_call_one(model_name, model_cfg, api_keys, system, user, pass_name=""):
            called.append(model_name)
            return {"content": f"{model_name}_result", "failed": False, "tokens": {}, "elapsed": 0.1, "model": model_name}

        with patch("callers.call_one", side_effect=_fake_call_one):
            results = call_all(
                system_prompt="system",
                user_prompt="user",
                user_config=_MOCK_USER_CONFIG,
                models=["claude"],
                pass_name="test",
            )

        assert called == ["claude"]
        assert "openai" not in results
        assert "gemini" not in results

    def test_call_all_excludes_perplexity(self):
        """Perplexity is excluded by default."""
        from callers import call_all, clear_api_call_log
        clear_api_call_log()

        config_with_perplexity = {
            **_MOCK_USER_CONFIG,
            "models": {
                **_MOCK_USER_CONFIG["models"],
                "perplexity": {"provider": "perplexity", "model": "sonar"},
            },
            "api_keys": {
                **_MOCK_USER_CONFIG["api_keys"],
                "perplexity": {"api_key": "pplx-test"},
            },
        }

        called = []

        def _fake_call_one(model_name, *a, **kw):
            called.append(model_name)
            return {"content": "", "failed": False, "tokens": {}, "elapsed": 0.1, "model": model_name}

        with patch("callers.call_one", side_effect=_fake_call_one):
            call_all(system_prompt="s", user_prompt="u", user_config=config_with_perplexity)

        assert "perplexity" not in called
