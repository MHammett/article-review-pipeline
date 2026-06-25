"""Collector registry — auto-discovers built-in and custom collectors."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from .base import Collector, ConfigError

# Built-in collectors (imported explicitly)
from .wordpress import WordPressCollector
from .twitter import TwitterCollector
from .gmail import GmailCollector
from .outlook365 import Outlook365Collector
from .textfiles import TextFilesCollector

log = logging.getLogger(__name__)

_BUILTINS: list[type[Collector]] = [
    WordPressCollector,
    TwitterCollector,
    GmailCollector,
    Outlook365Collector,
    TextFilesCollector,
]


def _build_registry() -> dict[str, type[Collector]]:
    registry: dict[str, type[Collector]] = {}

    for cls in _BUILTINS:
        name = cls.SOURCE_NAME
        if name in registry:
            raise ConfigError(
                name,
                message=f"Duplicate SOURCE_NAME {name!r}: {registry[name].__name__} vs {cls.__name__}",
            )
        registry[name] = cls

    custom_dir = Path(__file__).parent / "custom"
    if custom_dir.exists():
        for py_file in sorted(custom_dir.glob("*.py")):
            if py_file.stem.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"ci_web_intel.collectors.custom.{py_file.stem}", py_file
                )
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules[f"ci_web_intel.collectors.custom.{py_file.stem}"] = mod
                spec.loader.exec_module(mod)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    try:
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, Collector)
                            and attr is not Collector
                            and attr.SOURCE_NAME
                        ):
                            name = attr.SOURCE_NAME
                            if name in registry:
                                raise ConfigError(
                                    name,
                                    message=f"Duplicate SOURCE_NAME {name!r}: {registry[name].__name__} vs {attr.__name__} (in {py_file})",
                                )
                            registry[name] = attr
                            log.debug(
                                "Loaded custom collector %r from %s", name, py_file
                            )
                    except ConfigError:
                        raise
                    except Exception:
                        pass
            except ConfigError:
                raise
            except Exception as e:
                log.warning("Failed to load custom collector from %s: %s", py_file, e)

    return registry


REGISTRY: dict[str, type[Collector]] = _build_registry()
