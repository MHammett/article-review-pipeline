"""Mistral adapter."""

from __future__ import annotations

from ..config import ProviderConfig
from ._base import _with_retry

_DEFAULT_MODEL = "mistral-large-latest"

# Two-level try/except: outer guards the main SDK import, inner guards the
# rate-limit exception class.  ``SDKError`` lives in ``mistralai.models`` and
# was added in a later SDK release — the inner ImportError keeps us compatible
# with older pins that don't have it yet.  When ``_MistralRateLimit`` is None,
# ``_with_retry`` treats every exception as fatal (no retries), which is a safe
# degraded behaviour.
try:
    from mistralai import Mistral

    _available = True
    try:
        from mistralai.models import (
            SDKError as _MistralRateLimit,  # rate-limit manifests as SDKError 429
        )
    except ImportError:
        _MistralRateLimit = None  # type: ignore[assignment,misc]
except ImportError:
    _available = False
    Mistral = None  # type: ignore[assignment,misc]
    _MistralRateLimit = None  # type: ignore[assignment,misc]


class MistralAdapter:
    """LLM adapter for Mistral AI models.

    Uses the ``mistralai`` SDK (optional dep: ``ci-core[mistral]``).
    Config is read from ``LLMSettings.mistral`` (a ``ProviderConfig``):
    ``api_key``, ``model`` (defaults to ``mistral-large-latest``), ``timeout``,
    and ``max_retries``.

    Rate-limit errors in the Mistral SDK surface as ``SDKError`` with HTTP
    status 429.  When ``mistralai.models.SDKError`` is not importable (older
    SDK versions), rate-limit retries are disabled and the error propagates
    immediately.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialise the Mistral async client.

        Args:
            config: Provider config from ``LLMSettings.mistral``.

        Raises:
            RuntimeError: If the ``mistralai`` package is not installed.
        """
        if not _available:
            raise RuntimeError(
                "mistralai package not installed; pip install 'ci-core[mistral]'"
            )
        self._client = Mistral(api_key=config.api_key)  # type: ignore[misc]
        self._model = config.model or _DEFAULT_MODEL
        self._max_attempts = config.max_retries

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send *prompt* to Mistral and return the response text.

        When *system* is non-empty it is prepended as a ``{"role": "system"}``
        message, consistent with the OpenAI chat-completions convention that
        Mistral's API also follows.
        """
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
