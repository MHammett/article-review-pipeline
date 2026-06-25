"""Configurable modular logging setup for the voice profile bootstrap tool."""

from __future__ import annotations

import json
import logging
import logging.config
import sys
from datetime import datetime, timezone
from typing import Any


class _HumanFormatter(logging.Formatter):
    FMT = "%(asctime)s %(levelname)-7s %(name)-40s %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        self.datefmt = self.DATEFMT
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include structured extra fields at the top level
        skip = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "message",
        }
        for key, val in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                base[key] = val
        try:
            return json.dumps(base, default=str)
        except Exception:
            return json.dumps(base, default=repr)


def configure_logging(
    logging_cfg: dict | None = None, log_level_override: str | None = None
) -> None:
    """Configure logging from the sources.yaml `logging:` block.

    Call once at startup before any module logs. A CLI --log-level override
    takes precedence over the config-level setting.
    """
    cfg = logging_cfg or {}
    level_str = (log_level_override or cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)
    fmt = cfg.get("format", "human")
    log_file = cfg.get("file")
    also_stdout = cfg.get("also_stdout", True)
    module_overrides: dict[str, str] = cfg.get("modules") or {}

    formatter = (
        _JsonFormatter()
        if fmt == "json"
        else _HumanFormatter(_HumanFormatter.FMT, _HumanFormatter.DATEFMT)
    )

    handlers: list[logging.Handler] = []

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        handlers.append(fh)

    if not log_file or also_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        handlers.append(sh)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicates on re-configure
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)

    # Apply per-module overrides
    for module_name, module_level_str in module_overrides.items():
        module_level = getattr(logging, module_level_str.upper(), level)
        logging.getLogger(module_name).setLevel(module_level)
