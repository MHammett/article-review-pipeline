"""Shared outbound-HTTP constants for the Content Intelligence platform.

The User-Agent is a *platform* identifier (per docs/NAMING.md): outbound HTTP
calls from any package present the brand `content-intelligence/<version>`, not a
per-component name. Defining it once here keeps every caller in sync and prevents
the string from drifting back to per-package values.
"""

from importlib import metadata

try:
    _VERSION = metadata.version("ci-core")
except metadata.PackageNotFoundError:  # pragma: no cover - source/dev checkout
    _VERSION = "0.1.0"

#: Outbound HTTP User-Agent shared by all Content Intelligence packages.
USER_AGENT = f"content-intelligence/{_VERSION}"

__all__ = ["USER_AGENT"]
