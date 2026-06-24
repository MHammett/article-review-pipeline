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
    @staticmethod
    def get(provider: str, llm_settings: LLMSettings | None = None) -> Adapter:
        """Return an Adapter instance for *provider*.

        Reads provider config from *llm_settings*, defaulting to the global
        Settings singleton when not supplied.
        """
        if provider not in _REGISTRY:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. Choose from: {sorted(_REGISTRY)}"
            )
        if llm_settings is None:
            llm_settings = get_settings().llm
        config = getattr(llm_settings, provider)
        return _REGISTRY[provider](config)  # type: ignore[return-value]
