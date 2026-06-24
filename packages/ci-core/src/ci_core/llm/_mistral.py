"""Mistral adapter."""

from __future__ import annotations

from ..config import ProviderConfig
from ._base import _with_retry

_DEFAULT_MODEL = "mistral-large-latest"

try:
    from mistralai import Mistral

    _available = True
    try:
        from mistralai.models import (
            SDKError as _MistralRateLimit,
        )  # rate-limit manifests as SDKError 429
    except ImportError:
        _MistralRateLimit = None  # type: ignore[assignment,misc]
except ImportError:
    _available = False
    Mistral = None  # type: ignore[assignment,misc]
    _MistralRateLimit = None  # type: ignore[assignment,misc]


class MistralAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        if not _available:
            raise RuntimeError(
                "mistralai package not installed; pip install 'ci-core[mistral]'"
            )
        self._client = Mistral(api_key=config.api_key)
        self._model = config.model or _DEFAULT_MODEL
        self._max_attempts = config.max_retries

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        rate_limit_excs = (_MistralRateLimit,) if _MistralRateLimit else ()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async def _call() -> str:
            response = await self._client.chat.complete_async(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        return await _with_retry(
            _call, max_attempts=self._max_attempts, rate_limit_excs=rate_limit_excs
        )
