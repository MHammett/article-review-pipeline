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
    """LLM adapter for Google Gemini models.

    Uses the ``google-generativeai`` SDK (optional dep: ``ci-core[gemini]``).
    Config is read from ``LLMSettings.gemini`` (a ``ProviderConfig``):
    ``api_key``, ``model`` (defaults to ``gemini-1.5-pro``), ``timeout``,
    and ``max_retries``.

    Rate-limit errors surface as ``google.api_core.exceptions.ResourceExhausted``
    and are retried via ``_with_retry``.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialise the Gemini adapter.

        Calls ``genai.configure(api_key=...)`` to set the API key.  This is a
        *module-global side effect* in the ``google-generativeai`` SDK — it
        configures the default credentials for all subsequent SDK calls in the
        process.  Instantiating multiple ``GeminiAdapter`` objects with
        different keys will overwrite each other's configuration; use a single
        instance per process.

        Args:
            config: Provider config from ``LLMSettings.gemini``.

        Raises:
            RuntimeError: If the ``google-generativeai`` package is not
                installed.
        """
        if not _available:
            raise RuntimeError(
                "google-generativeai package not installed; pip install 'ci-core[gemini]'"
            )
        genai.configure(api_key=config.api_key)  # type: ignore[union-attr]
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
        """Send *prompt* to Gemini and return the response text.

        When *system* is non-empty it is passed as ``system_instruction`` to
        ``GenerativeModel``; an empty string is converted to ``None`` so the
        SDK omits the field entirely (passing an empty string can cause
        validation errors in some model versions).

        A new ``GenerativeModel`` instance is created per call because the SDK
        bundles ``system_instruction`` into the model object rather than
        accepting it per-request.
        """
        rate_limit_excs = (_GeminiRateLimit,) if _GeminiRateLimit is not None else ()
        model = genai.GenerativeModel(  # type: ignore[union-attr]
            model_name=self._model_name,
            system_instruction=system or None,
        )
        generation_config = genai.GenerationConfig(  # type: ignore[union-attr]
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
