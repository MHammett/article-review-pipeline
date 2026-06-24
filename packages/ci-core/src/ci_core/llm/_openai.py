"""OpenAI GPT adapter (also base for Grok, which uses the OpenAI-compatible API)."""

from __future__ import annotations

from ..config import ProviderConfig
from ._base import _with_retry

_DEFAULT_MODEL = "gpt-4o"

try:
    from openai import AsyncOpenAI as AsyncOpenAI
    from openai import RateLimitError as _OpenAIRateLimit

    _available = True
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    _OpenAIRateLimit = None  # type: ignore[assignment,misc]
    _available = False


class OpenAIAdapter:
    def __init__(self, config: ProviderConfig, *, base_url: str | None = None) -> None:
        if not _available:
            raise RuntimeError(
                "openai package not installed; pip install 'ci-core[openai]'"
            )
        self._client = AsyncOpenAI(  # type: ignore[misc]
            api_key=config.api_key,
            timeout=float(config.timeout),
            max_retries=0,
            **({"base_url": base_url} if base_url else {}),
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
        rate_limit_excs = (_OpenAIRateLimit,) if _OpenAIRateLimit else ()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async def _call() -> str:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        return await _with_retry(
            _call, max_attempts=self._max_attempts, rate_limit_excs=rate_limit_excs
        )
