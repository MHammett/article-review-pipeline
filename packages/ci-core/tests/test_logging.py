"""Tests for ci_core.logging.

JSON-shape and context-binding tests configure structlog with JSONRenderer and
no caching, then capture stdout via capsys to verify actual rendered output.
capture_logs() is used only for the middleware handler-context test.
"""

from __future__ import annotations

import pytest

# ci_core.config/.db/.models/.logging moved behind the `persistence` extra
# (audit finding 13): they have no production consumer, and making them
# mandatory meant every CLI user installed an async PostgreSQL driver and an
# ASGI web framework for a tool that never serves HTTP. Skip cleanly when the
# extra is absent; CI installs it, so coverage is unchanged there.
pytest.importorskip(
    "structlog",
    reason="ci-core[persistence] not installed — uv sync --extra persistence",
)


import json
import logging

import pytest
import structlog
import structlog.contextvars
import structlog.dev
import structlog.processors
from ci_core.config import AppSettings
from ci_core.logging import (
    LoggingMiddleware,
    bind_package_name,
    bind_request_id,
    bind_task_id,
    clear_context,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _reset_structlog():
    """Clear contextvars and reset structlog to safe defaults before each test."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


# ---------------------------------------------------------------------------
# Renderer selection
# ---------------------------------------------------------------------------


def test_configure_logging_production_uses_json_renderer():
    configure_logging(AppSettings(env="production"))
    processors = structlog.get_config()["processors"]
    assert any(isinstance(p, structlog.processors.JSONRenderer) for p in processors)


def test_configure_logging_development_uses_console_renderer():
    configure_logging(AppSettings(env="development"))
    processors = structlog.get_config()["processors"]
    assert any(isinstance(p, structlog.dev.ConsoleRenderer) for p in processors)


def test_configure_logging_default_is_development():
    """AppSettings defaults env to 'development', so no args → ConsoleRenderer."""
    configure_logging()
    processors = structlog.get_config()["processors"]
    assert any(isinstance(p, structlog.dev.ConsoleRenderer) for p in processors)


def test_configure_logging_staging_uses_console_renderer():
    configure_logging(AppSettings(env="staging"))
    processors = structlog.get_config()["processors"]
    assert any(isinstance(p, structlog.dev.ConsoleRenderer) for p in processors)


# ---------------------------------------------------------------------------
# JSON output shape
# ---------------------------------------------------------------------------


def _production_configure() -> None:
    """Configure structlog with production (JSON) renderer, no caching."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def test_json_output_required_keys(capsys):
    """Each log line must be valid JSON with event, level, and timestamp."""
    _production_configure()

    get_logger("test").info("hello world")

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["event"] == "hello world"
    assert payload["level"] == "info"
    assert "timestamp" in payload


def test_json_output_extra_kwargs(capsys):
    """Extra keyword arguments appear as top-level keys in the JSON object."""
    _production_configure()

    get_logger("test").info("check", foo="bar", count=42)

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["foo"] == "bar"
    assert payload["count"] == 42


def test_json_output_named_logger(capsys):
    """get_logger(name) binds a 'logger' key visible in the JSON output."""
    _production_configure()

    get_logger("my-component").info("named")

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["logger"] == "my-component"


# ---------------------------------------------------------------------------
# Context binding
# ---------------------------------------------------------------------------


def test_bind_request_id_appears_in_log(capsys):
    _production_configure()
    bind_request_id("req-abc123")
    get_logger().info("with request id")
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["request_id"] == "req-abc123"


def test_bind_task_id_appears_in_log(capsys):
    _production_configure()
    bind_task_id("task-xyz789")
    get_logger().info("with task id")
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["task_id"] == "task-xyz789"


def test_bind_package_name_appears_in_log(capsys):
    _production_configure()
    bind_package_name("ci-style-profile")
    get_logger().info("with package")
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["package_name"] == "ci-style-profile"


def test_all_three_context_vars(capsys):
    _production_configure()
    bind_request_id("r1")
    bind_task_id("t1")
    bind_package_name("ci-core")
    get_logger().info("all bound")
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["request_id"] == "r1"
    assert payload["task_id"] == "t1"
    assert payload["package_name"] == "ci-core"


def test_clear_context_removes_bound_fields(capsys):
    _production_configure()
    bind_request_id("req-gone")
    clear_context()
    get_logger().info("after clear")
    payload = json.loads(capsys.readouterr().out.strip())
    assert "request_id" not in payload


def test_context_does_not_leak_between_calls(capsys):
    _production_configure()
    bind_request_id("req-A")
    get_logger().info("first")
    capsys.readouterr()  # discard first output
    clear_context()
    bind_request_id("req-B")
    get_logger().info("second")
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["request_id"] == "req-B"


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------


def test_logging_middleware_is_importable():
    assert LoggingMiddleware is not None


def test_logging_middleware_sets_request_id_header():
    """Middleware must set X-Request-ID on every response."""
    import uuid

    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", homepage)])
    app.add_middleware(LoggingMiddleware)

    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    uuid.UUID(response.headers["x-request-id"])  # must be a valid UUID


def test_logging_middleware_binds_request_id_to_context():
    """request_id bound by middleware must be visible in contextvars within handlers."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    context_in_handler: dict = {}

    def handler(request: Request) -> PlainTextResponse:
        context_in_handler.update(structlog.contextvars.get_contextvars())
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", handler)])
    app.add_middleware(LoggingMiddleware)

    TestClient(app).get("/")
    assert "request_id" in context_in_handler
