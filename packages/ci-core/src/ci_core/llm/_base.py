"""LLM adapter base: Adapter Protocol, retry utility, AdapterError."""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, Protocol, runtime_checkable


class AdapterError(Exception):
    """Raised when an LLM adapter call fails after all retries."""


@runtime_checkable
class Adapter(Protocol):
    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str: ...


async def _with_retry(
    coro_fn: Callable[[], Awaitable[str]],
    *,
    max_attempts: int = 3,
    rate_limit_excs: tuple[type[BaseException], ...] = (),
) -> str:
    """Call coro_fn up to max_attempts times; retry on rate_limit_excs with exponential backoff."""
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
