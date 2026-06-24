"""Anthropic Claude adapter."""

from __future__ import annotations

from ..config import ProviderConfig
from ._base import _with_retry

_DEFAULT_MODEL = "claude-sonnet-4-6"

# ``as AsyncAnthropic`` (same name) makes the binding re-exportable and
# visible to mypy even though the import is inside a try block.
try:
    from anthropic import AsyncAnthropic as AsyncAnthropic
    from anthropic import RateLimitError as _AnthropicRateLimit

    _available = True
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment,misc]
    _AnthropicRateLimit = None  # type: ignore[assignment,misc]
    _available = False


class AnthropicAdapter:
    """LLM adapter for Anthropic Claude models.

    Uses the ``anthropic`` SDK (optional dep: ``ci-core[anthropic]``).
    Config is read from ``LLMSettings.anthropic`` (a ``ProviderConfig``):
    ``api_key``, ``model`` (defaults to ``claude-sonnet-4-6``), ``timeout``,
    and ``max_retries`` (controls how many times rate-limit errors are retried).
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialise the Anthropic async client.

        Args:
            config: Provider config from ``LLMSettings.anthropic``.

        Raises:
            RuntimeError: If the ``anthropic`` package is not installed.
        """
        if not _available:
            raise RuntimeError(
                "anthropic package not installed; pip install 'ci-core[anthropic]'"
            )
        self._client = AsyncAnthropic(  # type: ignore[misc]
            api_key=config.api_key,
            timeout=float(config.timeout),
            max_retries=0,  # retry logic lives in _with_retry, not the SDK
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
        """Send *prompt* to Claude and return the response text.

        The ``system`` parameter is omitted from the API call entirely when
        empty — Anthropic's API treats a missing ``system`` field differently
        from an empty string in some model versions.
        """
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
