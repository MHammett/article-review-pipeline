"""xAI Grok adapter — OpenAI-compatible API at api.x.ai."""

from __future__ import annotations

from ..config import ProviderConfig
from ._openai import OpenAIAdapter

_GROK_BASE_URL = "https://api.x.ai/v1"
_DEFAULT_MODEL = "grok-2"


class GrokAdapter(OpenAIAdapter):
    def __init__(self, config: ProviderConfig) -> None:
        effective = config.model_copy(update={"model": config.model or _DEFAULT_MODEL})
        super().__init__(effective, base_url=_GROK_BASE_URL)
