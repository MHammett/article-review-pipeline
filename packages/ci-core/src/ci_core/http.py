"""Shared outbound-HTTP constants for the Content Intelligence platform.

The User-Agent is a *platform* identifier (per docs/NAMING.md): outbound HTTP
calls from any package present the brand `content-intelligence/<version>`, not a
per-component name. Defining it once here keeps every caller in sync and prevents
the string from drifting back to per-package values.
"""

import ipaddress
import socket
from importlib import metadata
from urllib.parse import urlparse

import requests

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


# ---------------------------------------------------------------------------
# SSRF-guarded fetching
# ---------------------------------------------------------------------------
#
# This lives in ci_core rather than in ci-article-review's analysis/links.py,
# where the guard originally sat, for one concrete reason: adapters/citation/
# could not import from analysis/ without closing an import cycle, so the
# citation resolver fetched model-supplied URLs with no validation at all while
# user-supplied URLs were checked. The trust ordering was exactly inverted, and
# the cause was module placement. Putting the guard where every package can
# reach it makes the safe call the convenient one.

#: Redirect hops followed before giving up. Each hop is re-validated.
_MAX_REDIRECTS = 5


class UnsafeURLError(ValueError):
    """Raised when a URL resolves to a non-public address, or redirects to one."""


def is_public_host(url, *, fail_open_on_dns_error=False):
    """True if ``url``'s host resolves only to public, routable addresses.

    Rejects loopback, private, link-local and other non-global addresses —
    which covers the cloud-metadata endpoint (169.254.169.254), localhost, and
    anything on the LAN.

    ``fail_open_on_dns_error`` decides what an unresolvable hostname means.
    Link *validation* wants True: the goal there is an accurate error message
    for the author, and letting the HTTP layer produce the real DNS error beats
    reporting it as a security refusal. Anything that *fetches and then uses*
    the body wants the default, False — a name that fails ``getaddrinfo`` here
    but resolves through some other path at request time is precisely the
    interesting case.

    Not airtight on its own: DNS can change between this check and the socket
    connect (rebinding). ``safe_get`` narrows that window by re-validating each
    redirect hop; closing it entirely would mean connecting to a pinned IP with
    an explicit Host header, which is more machinery than this threat model
    needs.
    """
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return bool(fail_open_on_dns_error)
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if not ip.is_global or ip.is_loopback or ip.is_private or ip.is_link_local:
            return False
    return True


def safe_get(url, *, timeout=15, headers=None, allow_redirects=True, **kwargs):
    """``requests.get`` with the SSRF guard applied to every hop.

    Automatic redirect following is disabled and re-implemented so each
    ``Location`` is validated before it is followed. Plain
    ``allow_redirects=True`` validates the URL you pass and then follows an
    attacker-chosen chain unchecked, which makes the initial check decorative.

    Raises ``UnsafeURLError`` if the target — or any hop — is non-public, or if
    the chain exceeds ``_MAX_REDIRECTS``. Every other failure is a normal
    ``requests`` exception, so callers keep their existing error handling.
    """
    headers = headers if headers is not None else DEFAULT_HEADERS
    current = url
    for _hop in range(_MAX_REDIRECTS + 1):
        if not is_public_host(current):
            raise UnsafeURLError(
                f"Refusing to fetch a non-public/internal host (SSRF guard): {current}"
            )
        resp = requests.get(
            current, timeout=timeout, headers=headers, allow_redirects=False, **kwargs
        )
        if not allow_redirects or not resp.is_redirect:
            return resp
        location = resp.headers.get("Location")
        if not location:
            return resp
        # Relative Locations are legal and common; resolve before validating.
        current = requests.compat.urljoin(current, location)
    raise UnsafeURLError(f"Too many redirects (>{_MAX_REDIRECTS}) starting at {url}")


def safe_head(url, *, timeout=15, headers=None, **kwargs):
    """``requests.head`` behind the same guard, redirects re-validated per hop.

    Returns ``(response, final_url)`` — callers checking link health need to
    report where a redirect actually landed.
    """
    headers = headers if headers is not None else DEFAULT_HEADERS
    current = url
    for _hop in range(_MAX_REDIRECTS + 1):
        if not is_public_host(current):
            raise UnsafeURLError(
                f"Refusing to fetch a non-public/internal host (SSRF guard): {current}"
            )
        resp = requests.head(
            current, timeout=timeout, headers=headers, allow_redirects=False, **kwargs
        )
        if not resp.is_redirect:
            return resp, current
        location = resp.headers.get("Location")
        if not location:
            return resp, current
        current = requests.compat.urljoin(current, location)
    raise UnsafeURLError(f"Too many redirects (>{_MAX_REDIRECTS}) starting at {url}")


__all__ = [
    "USER_AGENT",
    "DEFAULT_HEADERS",
    "UnsafeURLError",
    "is_public_host",
    "safe_get",
    "safe_head",
]
