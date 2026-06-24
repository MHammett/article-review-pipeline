"""AdapterFactory: maps provider name -> Adapter instance."""

from __future__ import annotations

from ..config import LLMSettings, get_settings
from ._base import Adapter
from ._anthropic import AnthropicAdapter
from ._gemini import GeminiAdapter
from ._grok import GrokAdapter
from ._mistral import MistralAdapter
from ._openai import OpenAIAdapter

_REGISTRY: dict[str, type] = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "mistral": MistralAdapter,
    "grok": GrokAdapter,
}


class AdapterFactory:
    """Creates ``Adapter`` instances by provider name.

    Provider config is read from a ``LLMSettings`` object (which in turn is
    read from environment variables or a config file via ``get_settings()``).
    Each entry in ``LLMSettings`` is a ``ProviderConfig`` named after its
    provider, so ``getattr(llm_settings, "anthropic")`` returns the
    ``ProviderConfig`` for Anthropic — no lookup table needed.
    """

    @staticmethod
    def get(provider: str, llm_settings: LLMSettings | None = None) -> Adapter:
        """Return a configured ``Adapter`` instance for *provider*.

        Resolves the correct ``ProviderConfig`` from *llm_settings* using
        ``getattr(llm_settings, provider)`` — this works because
        ``LLMSettings`` has one attribute per provider, named to match the
        registry keys (``"anthropic"``, ``"openai"``, ``"gemini"``,
        ``"mistral"``, ``"grok"``).

        Args:
            provider: Provider name.  Must be one of the keys in ``_REGISTRY``:
                ``"anthropic"``, ``"openai"``, ``"gemini"``, ``"mistral"``,
                or ``"grok"``.
            llm_settings: Optional settings object.  Defaults to the global
                ``Settings`` singleton (``get_settings().llm``) when ``None``.

        Returns:
            A concrete adapter instance that satisfies the ``Adapter`` protocol.

        Raises:
            ValueError: If *provider* is not a recognised provider name.
            RuntimeError: If the provider's optional SDK package is not
                installed (propagated from the adapter's ``__init__``).
        """
        if provider not in _REGISTRY:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. Choose from: {sorted(_REGISTRY)}"
            )
        if llm_settings is None:
            llm_settings = get_settings().llm
        config = getattr(llm_settings, provider)
        return _REGISTRY[provider](config)  # type: ignore[return-value]
