"""Structured logging for CI tools using structlog.

Production mode: JSON renderer. Development mode: ConsoleRenderer.
Mode is detected via AppSettings.env ("production" → JSON, anything else → console).

Usage::

    from ci_core.logging import configure_logging, get_logger

    configure_logging()           # call once at app startup
    log = get_logger(__name__)
    log.info("started", version="1.0")

Context variables (request_id, task_id, package_name) propagate automatically
via structlog.contextvars. Bind them with the provided helpers; call
clear_context() between unrelated units of work.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
import structlog.contextvars
import structlog.dev
import structlog.processors
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ci_core.config import AppSettings


def configure_logging(app_settings: AppSettings | None = None) -> None:
    """Configure structlog globally. Call once at application startup.

    Reads app_settings.env to select the renderer:
    - "production" → JSONRenderer (machine-readable, one line per event)
    - anything else → ConsoleRenderer (human-readable, coloured)
    """
    if app_settings is None:
        app_settings = AppSettings()

    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if app_settings.env == "production"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "") -> structlog.types.FilteringBoundLogger:
    """Return a structlog bound logger, optionally tagged with a logger name."""
    logger: structlog.types.FilteringBoundLogger = structlog.get_logger()
    if name:
        logger = logger.bind(logger=name)
    return logger


# ---------------------------------------------------------------------------
# Context-variable helpers
# ---------------------------------------------------------------------------


def bind_request_id(request_id: str) -> None:
    """Bind request_id to all log events in the current contextvars context."""
    structlog.contextvars.bind_contextvars(request_id=request_id)


def bind_task_id(task_id: str) -> None:
    """Bind task_id to all log events in the current contextvars context."""
    structlog.contextvars.bind_contextvars(task_id=task_id)


def bind_package_name(package_name: str) -> None:
    """Bind package_name to all log events in the current contextvars context."""
    structlog.contextvars.bind_contextvars(package_name=package_name)


def clear_context() -> None:
    """Clear all contextvars-bound log fields."""
    structlog.contextvars.clear_contextvars()


# ---------------------------------------------------------------------------
# FastAPI / Starlette middleware
# ---------------------------------------------------------------------------


class LoggingMiddleware(BaseHTTPMiddleware):
    """Per-request middleware that injects a unique request_id and logs outcomes.

    Binds request_id to the structlog contextvars context for the duration of
    the request so all log events within a request carry it automatically.
    Sets X-Request-ID on every response.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        t0 = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        get_logger("http").info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
