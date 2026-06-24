"""Tests for ci_core.llm — adapters, retry logic, and factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci_core.config import LLMSettings, ProviderConfig
from ci_core.llm._base import AdapterError, _with_retry


# ---------------------------------------------------------------------------
# _with_retry
# ---------------------------------------------------------------------------


class _FakeRateLimit(Exception):
    pass


@pytest.mark.asyncio
async def test_retry_succeeds_first_attempt():
    called = 0

    async def _ok():
        nonlocal called
        called += 1
        return "ok"

    result = await _with_retry(_ok, max_attempts=3)
    assert result == "ok"
    assert called == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_rate_limit(monkeypatch):
    monkeypatch.setattr("ci_core.llm._base.asyncio.sleep", AsyncMock())
    attempts = 0

    async def _flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _FakeRateLimit("throttled")
        return "done"

    result = await _with_retry(
        _flaky, max_attempts=3, rate_limit_excs=(_FakeRateLimit,)
    )
    assert result == "done"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_raises_adapter_error_after_exhaustion(monkeypatch):
    monkeypatch.setattr("ci_core.llm._base.asyncio.sleep", AsyncMock())

    async def _always_limited():
        raise _FakeRateLimit("always")

    with pytest.raises(AdapterError):
        await _with_retry(
            _always_limited, max_attempts=3, rate_limit_excs=(_FakeRateLimit,)
        )


@pytest.mark.asyncio
async def test_retry_reraises_non_rate_limit_immediately():
    async def _bad():
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        await _with_retry(_bad, max_attempts=3, rate_limit_excs=(_FakeRateLimit,))


# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------


def _make_anthropic_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


@pytest.mark.asyncio
async def test_anthropic_adapter_complete():
    config = ProviderConfig(api_key="sk-test", model="claude-sonnet-4-6")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        return_value=_make_anthropic_response("Hello!")
    )

    with (
        patch("ci_core.llm._anthropic._available", True),
        patch("ci_core.llm._anthropic.AsyncAnthropic", return_value=mock_client),
    ):
        from ci_core.llm._anthropic import AnthropicAdapter

        adapter = AnthropicAdapter(config)
        adapter._client = mock_client
        result = await adapter.complete("Hi", system="Be helpful")

    assert result == "Hello!"
    mock_client.messages.create.assert_awaited_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "Be helpful"
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
async def test_anthropic_adapter_no_system():
    config = ProviderConfig(api_key="sk-test")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_make_anthropic_response("Hi"))

    with (
        patch("ci_core.llm._anthropic._available", True),
        patch("ci_core.llm._anthropic.AsyncAnthropic", return_value=mock_client),
    ):
        from ci_core.llm._anthropic import AnthropicAdapter

        adapter = AnthropicAdapter(config)
        adapter._client = mock_client
        await adapter.complete("Hi")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "system" not in call_kwargs


# ---------------------------------------------------------------------------
# OpenAIAdapter
# ---------------------------------------------------------------------------


def _make_openai_response(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_openai_adapter_complete():
    config = ProviderConfig(api_key="sk-test", model="gpt-4o")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("Pong")
    )

    with (
        patch("ci_core.llm._openai._available", True),
        patch("ci_core.llm._openai.AsyncOpenAI", return_value=mock_client),
    ):
        from ci_core.llm._openai import OpenAIAdapter

        adapter = OpenAIAdapter(config)
        adapter._client = mock_client
        result = await adapter.complete("Ping", system="Respond with Pong")

    assert result == "Pong"
    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "Respond with Pong"}
    assert messages[1] == {"role": "user", "content": "Ping"}


# ---------------------------------------------------------------------------
# AdapterFactory
# ---------------------------------------------------------------------------


def test_factory_unknown_provider():
    from ci_core.llm._factory import AdapterFactory

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        AdapterFactory.get("nonexistent", llm_settings=LLMSettings())


def test_factory_returns_anthropic():
    from ci_core.llm._factory import AdapterFactory
    from ci_core.llm._anthropic import AnthropicAdapter

    settings = LLMSettings(anthropic=ProviderConfig(api_key="sk-x"))

    with (
        patch("ci_core.llm._anthropic._available", True),
        patch("ci_core.llm._anthropic.AsyncAnthropic"),
    ):
        adapter = AdapterFactory.get("anthropic", llm_settings=settings)

    assert isinstance(adapter, AnthropicAdapter)


def test_factory_returns_openai():
    from ci_core.llm._factory import AdapterFactory
    from ci_core.llm._openai import OpenAIAdapter

    settings = LLMSettings(openai=ProviderConfig(api_key="sk-x"))

    with (
        patch("ci_core.llm._openai._available", True),
        patch("ci_core.llm._openai.AsyncOpenAI"),
    ):
        adapter = AdapterFactory.get("openai", llm_settings=settings)

    assert isinstance(adapter, OpenAIAdapter)


def test_factory_returns_grok():
    from ci_core.llm._factory import AdapterFactory
    from ci_core.llm._grok import GrokAdapter

    settings = LLMSettings(grok=ProviderConfig(api_key="xai-x"))

    with (
        patch("ci_core.llm._openai._available", True),
        patch("ci_core.llm._openai.AsyncOpenAI"),
    ):
        adapter = AdapterFactory.get("grok", llm_settings=settings)

    assert isinstance(adapter, GrokAdapter)


def test_grok_uses_xai_base_url():
    from ci_core.llm._grok import _GROK_BASE_URL, GrokAdapter

    settings = LLMSettings(grok=ProviderConfig(api_key="xai-x"))
    mock_openai_cls = MagicMock()

    with (
        patch("ci_core.llm._openai._available", True),
        patch("ci_core.llm._openai.AsyncOpenAI", mock_openai_cls),
    ):
        GrokAdapter(settings.grok)

    call_kwargs = mock_openai_cls.call_args.kwargs
    assert call_kwargs.get("base_url") == _GROK_BASE_URL


def test_grok_default_model():
    from ci_core.llm._grok import _DEFAULT_MODEL, GrokAdapter

    config_no_model = ProviderConfig(api_key="xai-x")

    with (
        patch("ci_core.llm._openai._available", True),
        patch("ci_core.llm._openai.AsyncOpenAI"),
    ):
        adapter = GrokAdapter(config_no_model)

    assert adapter._model == _DEFAULT_MODEL
