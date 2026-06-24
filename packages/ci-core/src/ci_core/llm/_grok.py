"""xAI Grok adapter — OpenAI-compatible API at api.x.ai."""

from __future__ import annotations

from ..config import ProviderConfig
from ._openai import OpenAIAdapter

_GROK_BASE_URL = "https://api.x.ai/v1"
_DEFAULT_MODEL = "grok-2"


class GrokAdapter(OpenAIAdapter):
    """LLM adapter for xAI Grok models.

    Grok exposes an OpenAI-compatible chat-completions API, so this adapter
    is a thin subclass of ``OpenAIAdapter`` that hard-wires the base URL to
    ``https://api.x.ai/v1`` and sets a Grok-appropriate default model.

    Uses the ``openai`` SDK (optional dep: ``ci-core[openai]``).
    Config is read from ``LLMSettings.grok`` (a ``ProviderConfig``):
    ``api_key``, ``model`` (defaults to ``grok-2``), ``timeout``,
    and ``max_retries``.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialise the Grok adapter.

        Applies the Grok default model via ``model_copy`` before delegating to
        ``OpenAIAdapter.__init__``.  ``model_copy`` (a Pydantic v2 method)
        returns a shallow copy of the config with the ``model`` field updated,
        so the original ``config`` object is not mutated — safe to call with a
        shared ``ProviderConfig`` instance.

        Args:
            config: Provider config from ``LLMSettings.grok``.

        Raises:
            RuntimeError: If the ``openai`` package is not installed.
        """
        effective = config.model_copy(update={"model": config.model or _DEFAULT_MODEL})
        super().__init__(effective, base_url=_GROK_BASE_URL)
