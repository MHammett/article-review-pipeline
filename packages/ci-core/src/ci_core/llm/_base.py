"""LLM adapter base: Adapter Protocol, retry utility, AdapterError."""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, Protocol, runtime_checkable


class AdapterError(Exception):
    """Raised when an LLM adapter call fails after all retries are exhausted.

    Only raised for rate-limit failures; other exceptions propagate immediately.
    """


@runtime_checkable
class Adapter(Protocol):
    """Provider-agnostic interface for a single LLM call.

    All adapters expose one method: ``complete``.  Callers never import a
    concrete adapter class directly — use ``AdapterFactory.get(provider)``
    instead so provider selection stays in config.
    """

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send *prompt* to the provider and return the text response.

        Args:
            prompt: The user-turn text.
            system: Optional system/instruction prompt.  Empty string means no
                system prompt (some providers omit the field entirely when
                empty rather than sending an empty string).
            temperature: Sampling temperature.  0 = deterministic, 1 = maximum
                randomness.  Defaults to 0.7.
            max_tokens: Upper bound on response tokens.  Defaults to 4096.

        Returns:
            The model's text response as a plain string.
        """
        ...


async def _with_retry(
    coro_fn: Callable[[], Awaitable[str]],
    *,
    max_attempts: int = 3,
    rate_limit_excs: tuple[type[BaseException], ...] = (),
) -> str:
    """Call *coro_fn* up to *max_attempts* times, retrying on rate-limit errors.

    Backoff formula: ``2^attempt + uniform(0, 1)`` seconds between retries.
    Attempt 0 → ~1 s, attempt 1 → ~2 s, attempt 2 raises AdapterError.

    Only exceptions that are instances of *rate_limit_excs* trigger a retry;
    all other exceptions propagate immediately without consuming retry budget.
    When *rate_limit_excs* is empty every exception propagates immediately.

    Args:
        coro_fn: Zero-argument async callable to call on each attempt.
        max_attempts: Maximum number of calls before giving up.
        rate_limit_excs: Tuple of exception types that should trigger a retry.

    Raises:
        AdapterError: When all *max_attempts* are exhausted on rate-limit errors.
        Exception: Any non-rate-limit exception from *coro_fn*, re-raised as-is.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await coro_fn()
        except Exception as exc:
            if not rate_limit_excs or not isinstance(exc, rate_limit_excs):
                raise
            last_exc = exc
            if attempt < max_attempts - 1:
                await asyncio.sleep((2**attempt) + random.uniform(0.0, 1.0))
    raise AdapterError(f"Rate-limited after {max_attempts} attempts") from last_exc
