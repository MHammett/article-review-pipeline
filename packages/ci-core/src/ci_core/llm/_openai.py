"""OpenAI GPT adapter (also base for Grok, which uses the OpenAI-compatible API)."""

from __future__ import annotations

from ..config import ProviderConfig
from ._base import _with_retry

_DEFAULT_MODEL = "gpt-4o"

# ``as AsyncOpenAI`` (same name) makes the binding re-exportable and
# visible to mypy even though the import is inside a try block.
try:
    from openai import AsyncOpenAI as AsyncOpenAI
    from openai import RateLimitError as _OpenAIRateLimit

    _available = True
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    _OpenAIRateLimit = None  # type: ignore[assignment,misc]
    _available = False


class OpenAIAdapter:
    """LLM adapter for OpenAI GPT models.

    Uses the ``openai`` SDK (optional dep: ``ci-core[openai]``).
    Config is read from ``LLMSettings.openai`` (a ``ProviderConfig``):
    ``api_key``, ``model`` (defaults to ``gpt-4o``), ``timeout``,
    and ``max_retries``.

    Also serves as the base for ``GrokAdapter``, which passes a custom
    *base_url* to redirect calls to xAI's OpenAI-compatible endpoint.
    """

    def __init__(self, config: ProviderConfig, *, base_url: str | None = None) -> None:
        """Initialise the OpenAI async client.

        Args:
            config: Provider config from ``LLMSettings.openai`` (or
                ``LLMSettings.grok`` when called from ``GrokAdapter``).
            base_url: Override the API base URL.  ``None`` uses the default
                OpenAI endpoint; ``GrokAdapter`` passes ``https://api.x.ai/v1``.

        Raises:
            RuntimeError: If the ``openai`` package is not installed.
        """
        if not _available:
            raise RuntimeError(
                "openai package not installed; pip install 'ci-core[openai]'"
            )
        self._client = AsyncOpenAI(  # type: ignore[misc]
            api_key=config.api_key,
            timeout=float(config.timeout),
            max_retries=0,  # retry logic lives in _with_retry, not the SDK
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
        """Send *prompt* to the model and return the response text.

        When *system* is non-empty it is prepended as a ``{"role": "system"}``
        message, which is the standard OpenAI chat-completions convention.
        """
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
