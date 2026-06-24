"""Anthropic Claude adapter."""

from __future__ import annotations

from ..config import ProviderConfig
from ._base import _with_retry

_DEFAULT_MODEL = "claude-sonnet-4-6"

try:
    from anthropic import AsyncAnthropic as AsyncAnthropic
    from anthropic import RateLimitError as _AnthropicRateLimit

    _available = True
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment,misc]
    _AnthropicRateLimit = None  # type: ignore[assignment,misc]
    _available = False


class AnthropicAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        if not _available:
            raise RuntimeError(
                "anthropic package not installed; pip install 'ci-core[anthropic]'"
            )
        self._client = AsyncAnthropic(  # type: ignore[misc]
            api_key=config.api_key,
            timeout=float(config.timeout),
            max_retries=0,  # we handle retries ourselves
        )
        self._model = config.model or _DEFAULT_MODEL
        self._max_attempts = config.max_retries

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        rate_limit_excs = (_AnthropicRateLimit,) if _AnthropicRateLimit else ()

        async def _call() -> str:
            kwargs: dict = dict(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            response = await self._client.messages.create(**kwargs)
            return response.content[0].text

        return await _with_retry(
            _call, max_attempts=self._max_attempts, rate_limit_excs=rate_limit_excs
        )
