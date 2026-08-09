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

# DEFAULT_HEADERS keeps the honest platform User-Agent (see docstring above —
# it's an intentional identity string, not something to spoof) but adds the
# Accept / Accept-Language headers a real browser always sends. A bare
# User-Agent-only request is itself a bot signal to some WAFs regardless of
# the UA string's contents; several of the 403s that motivated this constant
# came from sites that just wanted *a* plausible Accept header. This resolves
# some blocks without impersonating a browser, which is a smaller step than
# UA spoofing and carries a lot less risk of quietly evading a site's
# deliberate bot policy.
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

__all__ = ["USER_AGENT", "DEFAULT_HEADERS"]
