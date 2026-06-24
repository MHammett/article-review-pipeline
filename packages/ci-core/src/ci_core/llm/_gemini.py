"""Google Gemini adapter."""

from __future__ import annotations

from ..config import ProviderConfig
from ._base import _with_retry

_DEFAULT_MODEL = "gemini-1.5-pro"

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted as _GeminiRateLimit

    _available = True
except ImportError:
    _available = False
    genai = None  # type: ignore[assignment]
    _GeminiRateLimit = None  # type: ignore[assignment,misc]


class GeminiAdapter:
    def __init__(self, config: ProviderConfig) -> None:
        if not _available:
            raise RuntimeError(
                "google-generativeai package not installed; pip install 'ci-core[gemini]'"
            )
        genai.configure(api_key=config.api_key)
        self._model_name = config.model or _DEFAULT_MODEL
        self._config = config
        self._max_attempts = config.max_retries

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        rate_limit_excs = (_GeminiRateLimit,) if _GeminiRateLimit else ()
        model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system or None,
        )
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        async def _call() -> str:
            response = await model.generate_content_async(
                prompt, generation_config=generation_config
            )
            return response.text

        return await _with_retry(
            _call, max_attempts=self._max_attempts, rate_limit_excs=rate_limit_excs
        )
