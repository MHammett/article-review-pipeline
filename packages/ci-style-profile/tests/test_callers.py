"""Tests for callers.py — multi-model routing, SSE accumulation, error handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import litellm


# ---------------------------------------------------------------------------
# litellm-shaped stream mocks
# ---------------------------------------------------------------------------
# These patch the litellm call itself rather than the HTTP transport. The old
# versions mocked requests.Session, which after the litellm migration no longer
# intercepts anything — the calls went out to the real providers, and the
# failure-path tests "passed" because a real call fails too.


def _completion_stream(text: str):
    """A litellm.completion(stream=True) response carrying ``text`` and usage."""

    def chunk(content=None, finish_reason=None, usage=None):
        choice = SimpleNamespace(
            delta=SimpleNamespace(content=content), finish_reason=finish_reason
        )
        return SimpleNamespace(
            choices=[choice] if (content or finish_reason) else [],
            usage=usage,
            citations=None,
            search_results=None,
            vertex_ai_grounding_metadata=None,
        )

    return [
        chunk(content=text),
        chunk(finish_reason="stop"),
        chunk(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                prompt_tokens_details=None,
                cache_read_input_tokens=None,
            )
        ),
    ]


def _responses_stream(text: str):
    """A litellm.responses(stream=True) event stream — the OpenAI surface."""
    return [
        SimpleNamespace(type="response.output_text.delta", delta=text),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=20,
                    prompt_tokens_details=None,
                    cache_read_input_tokens=None,
                ),
                status="completed",
                incomplete_details=None,
            ),
        ),
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
        from ci_style_profile.callers import call_one, clear_api_call_log

        clear_api_call_log()

        with patch.object(
            litellm,
            "completion",
            return_value=_completion_stream('{"style_profile": "test"}'),
        ):
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
        assert '{"style_profile": "test"}' in result["content"]

    def test_anthropic_http_error(self):
        """call_one returns failed=True on a transport error; does not raise."""
        from ci_style_profile.callers import call_one, clear_api_call_log

        clear_api_call_log()

        with patch.object(
            litellm, "completion", side_effect=Exception("Connection refused")
        ):
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
        """OpenAI goes through responses(), not completion() — see ci_core.llm.client."""
        from ci_style_profile.callers import call_one, clear_api_call_log

        clear_api_call_log()

        with patch.object(
            litellm,
            "responses",
            return_value=_responses_stream('{"style_profile": "openai result"}'),
        ):
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
        from ci_style_profile.callers import call_one, clear_api_call_log

        clear_api_call_log()

        with patch.object(
            litellm, "responses", side_effect=Exception("503 Server Error")
        ):
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
        from ci_style_profile.callers import call_one, clear_api_call_log

        clear_api_call_log()

        with patch.object(
            litellm,
            "completion",
            return_value=_completion_stream('{"style_profile": "gemini result"}'),
        ):
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
        from ci_style_profile.callers import call_all, clear_api_call_log

        clear_api_call_log()

        def _fake_call_one(model_name, model_cfg, api_keys, system, user, pass_name=""):
            return {
                "content": f"{model_name}_result",
                "failed": False,
                "tokens": {},
                "elapsed": 0.1,
                "model": model_name,
            }

        with patch("ci_style_profile.callers.call_one", side_effect=_fake_call_one):
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
        from ci_style_profile.callers import call_all, clear_api_call_log

        clear_api_call_log()

        called = []

        def _fake_call_one(model_name, model_cfg, api_keys, system, user, pass_name=""):
            called.append(model_name)
            return {
                "content": f"{model_name}_result",
                "failed": False,
                "tokens": {},
                "elapsed": 0.1,
                "model": model_name,
            }

        with patch("ci_style_profile.callers.call_one", side_effect=_fake_call_one):
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

    def test_call_all_applies_wall_clock_backstop(self):
        """A call that overruns its budget is reported as a timeout, not awaited.

        Under streaming the socket timeout is only the inter-token read gap, so
        a model that keeps dribbling tokens needs a wall-clock bound on top.
        """
        import time

        from ci_style_profile.callers import call_all, clear_api_call_log

        clear_api_call_log()

        def _slow_call_one(model_name, *a, **kw):
            time.sleep(2)
            return {
                "content": "too late",
                "failed": False,
                "tokens": {},
                "elapsed": 2.0,
            }

        with (
            patch("ci_style_profile.callers.call_one", side_effect=_slow_call_one),
            patch(
                "ci_style_profile.callers.timeout_model.compute_all",
                return_value={"claude": 0.2},
            ),
        ):
            started = time.monotonic()
            results = call_all(
                system_prompt="s",
                user_prompt="u",
                user_config=_MOCK_USER_CONFIG,
                models=["claude"],
            )
            gave_up_after = time.monotonic() - started

        assert results["claude"]["failed"] is True
        assert "backstop" in results["claude"]["error"]
        assert results["claude"]["elapsed"] == 0.2
        # Reported back well before the call itself finished.
        assert gave_up_after < 1.5

    def test_call_all_uses_per_model_backstops_from_timeout_model(self):
        """Budgets come from the shared sliding-scale model, sized on prompt length."""
        from ci_style_profile.callers import call_all, clear_api_call_log

        clear_api_call_log()
        seen = {}

        def _capture(char_count, model_configs, ceiling, **kw):
            seen["char_count"] = char_count
            seen["models"] = sorted(model_configs)
            seen["ceiling"] = ceiling
            return {name: 30 for name in model_configs}

        def _fake_call_one(model_name, *a, **kw):
            return {"content": "", "failed": False, "tokens": {}, "elapsed": 0.0}

        with (
            patch("ci_style_profile.callers.call_one", side_effect=_fake_call_one),
            patch(
                "ci_style_profile.callers.timeout_model.compute_all",
                side_effect=_capture,
            ),
        ):
            call_all(
                system_prompt="s" * 100,
                user_prompt="u" * 400,
                user_config=_MOCK_USER_CONFIG,
            )

        assert seen["char_count"] == 500
        assert seen["models"] == ["claude", "gemini", "openai"]
        assert seen["ceiling"] > 0

    def test_call_all_excludes_perplexity(self):
        """Perplexity is excluded by default."""
        from ci_style_profile.callers import call_all, clear_api_call_log

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
            return {
                "content": "",
                "failed": False,
                "tokens": {},
                "elapsed": 0.1,
                "model": model_name,
            }

        with patch("ci_style_profile.callers.call_one", side_effect=_fake_call_one):
            call_all(
                system_prompt="s", user_prompt="u", user_config=config_with_perplexity
            )

        assert "perplexity" not in called
