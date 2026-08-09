"""Google Gemini adapter."""

from __future__ import annotations

from ..config import ProviderConfig
from ._base import _with_retry

_DEFAULT_MODEL = "gemini-1.5-pro"

# ``as Client``/``as ClientError`` (same name) makes the binding re-exportable
# and visible to mypy even though the import is inside a try block — mirrors
# the pattern in _anthropic.py so ``patch("ci_core.llm._gemini.Client", ...)``
# works the same way regardless of whether the SDK is installed.
try:
    from google.genai import Client as Client
    from google.genai.errors import ClientError as ClientError
    from google.genai.types import GenerateContentConfig as GenerateContentConfig

    _available = True
except ImportError:
    _available = False
    Client = None  # type: ignore[assignment,misc]
    ClientError = None  # type: ignore[assignment,misc]
    GenerateContentConfig = None  # type: ignore[assignment,misc]


class _GeminiRateLimit(Exception):
    """Raised in place of ``google.genai.errors.ClientError`` for HTTP 429.

    The SDK's ``ClientError`` covers the whole 4xx range with no dedicated
    rate-limit subclass, and ``_with_retry`` retries by exception *type* — so
    retrying on bare ``ClientError`` would also retry non-transient 4xx
    failures (bad request, auth). This subclass is raised only when
    ``ClientError.code == 429``, matching the old SDK's
    ``ResourceExhausted``-only retry behaviour.
    """


class GeminiAdapter:
    """LLM adapter for Google Gemini models.

    Uses the ``google-genai`` SDK (optional dep: ``ci-core[gemini]``).
    Config is read from ``LLMSettings.gemini`` (a ``ProviderConfig``):
    ``api_key``, ``model`` (defaults to ``gemini-1.5-pro``), ``timeout``,
    and ``max_retries``.

    Rate-limit errors (HTTP 429) are retried via ``_with_retry``; other
    ``ClientError``/``ServerError`` failures propagate immediately.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialise the Gemini adapter.

        Args:
            config: Provider config from ``LLMSettings.gemini``.

        Raises:
            RuntimeError: If the ``google-genai`` package is not installed.
        """
        if not _available:
            raise RuntimeError(
                "google-genai package not installed; pip install 'ci-core[gemini]'"
            )
        self._client = Client(api_key=config.api_key)  # type: ignore[misc]
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

        When *system* is non-empty it is passed as ``system_instruction``; an
        empty string is converted to ``None`` so the SDK omits the field
        entirely (passing an empty string can cause validation errors in some
        model versions).
        """
        generation_config = GenerateContentConfig(  # type: ignore[misc]
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        async def _call() -> str:
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=generation_config,
                )
            except ClientError as exc:  # type: ignore[misc]
                if exc.code == 429:
                    raise _GeminiRateLimit(str(exc)) from exc
                raise
            return response.text or ""

        return await _with_retry(
            _call, max_attempts=self._max_attempts, rate_limit_excs=(_GeminiRateLimit,)
        )
