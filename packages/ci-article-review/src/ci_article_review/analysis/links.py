"""URL extraction, HTTP status validation, and Wayback Machine archive checks."""

import concurrent.futures
import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse

import requests

from ci_core.http import USER_AGENT

from ..adapters.citation.wayback import check as wayback_check

log = logging.getLogger(__name__)

# Trailing punctuation characters that end a sentence but are not part of a URL.
_URL_RE = re.compile(r"https?://[^\s\)\]\>\"\'<,;]+")
_TRAILING_PUNCT = re.compile(r"[.,!?:;]+$")
_HEAD_TIMEOUT = 8
_MAX_PARALLEL = 10


def extract_urls(text):
    """Return deduplicated list of URLs found in text, trailing punctuation stripped."""
    raw = _URL_RE.findall(text)
    cleaned = [_TRAILING_PUNCT.sub("", u) for u in raw]
    return list(dict.fromkeys(u for u in cleaned if u))


def _is_public_host(url):
    """Return True if the URL's host resolves only to public (routable) addresses.

    Guards against SSRF: a draft (especially a third-party submission) could
    contain a link to cloud metadata (169.254.169.254), localhost, or another
    internal service.  We resolve the hostname and reject any result that is
    loopback, private, link-local, or otherwise non-global.  Not airtight
    against DNS-rebinding or a public URL that 302-redirects to an internal
    host, but it blocks the common cases cheaply.
    """
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        # DNS failure — let the HTTP layer handle it and report the error.
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not ip.is_global or ip.is_loopback or ip.is_private or ip.is_link_local:
            return False
    return True


def _check_http(url, timeout=_HEAD_TIMEOUT):
    """HEAD-check a single URL. Falls back to GET if HEAD returns 405."""
    if not _is_public_host(url):
        return {
            "status_code": None,
            "ok": False,
            "error": "skipped: non-public host (SSRF guard)",
        }
    try:
        resp = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        if resp.status_code == 405:
            with requests.get(
                url,
                allow_redirects=True,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
                stream=True,
            ) as resp:
                final_url = resp.url
                return {
                    "status_code": resp.status_code,
                    "ok": resp.status_code < 400,
                    "redirected_to": final_url if final_url != url else None,
                }
        final_url = resp.url
        return {
            "status_code": resp.status_code,
            "ok": resp.status_code < 400,
            "redirected_to": final_url if final_url != url else None,
        }
    except requests.exceptions.Timeout:
        return {"status_code": None, "ok": False, "error": "timeout"}
    except Exception as exc:
        return {"status_code": None, "ok": False, "error": str(exc)}


def _check_one(
    url, check_wayback, http_timeout, wayback_timeout, wayback_stale_days=None
):
    entry = {"url": url}
    http_result = _check_http(url, timeout=http_timeout)
    entry.update(http_result)
    # Skip Wayback for hosts the SSRF guard rejected — no point sending an
    # internal URL to archive.org, and it can't be publicly archived anyway.
    skipped_for_ssrf = "SSRF guard" in (http_result.get("error") or "")
    if check_wayback and not skipped_for_ssrf:
        entry["wayback"] = wayback_check(
            url, timeout=wayback_timeout, stale_days=wayback_stale_days
        )
    return entry


def validate_links(
    text,
    check_wayback=True,
    http_timeout=_HEAD_TIMEOUT,
    wayback_timeout=10,
    wayback_stale_days=None,
):
    """Extract and validate every URL found in text.

    Checks run in parallel (up to _MAX_PARALLEL threads) so a 15-URL article
    doesn't add 15× timeout to the run.

    wayback_stale_days overrides the default staleness threshold (180 days).
    Set via pipeline.wayback_snapshot_stale_days in user.yaml.

    Returns a list of dicts, one per unique URL:
      url           str   — the URL as found in the text (trailing punctuation stripped)
      status_code   int   — HTTP status code (None on network error)
      ok            bool  — True when status < 400
      redirected_to str   — final URL if a redirect occurred
      error         str   — set only on network error
      wayback       dict  — result from Wayback availability check (if check_wayback)
    """
    urls = extract_urls(text)
    if not urls:
        return []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(urls), _MAX_PARALLEL)
    ) as pool:
        futures = {
            pool.submit(
                _check_one,
                url,
                check_wayback,
                http_timeout,
                wayback_timeout,
                wayback_stale_days,
            ): url
            for url in urls
        }
        # Collect results preserving original URL order
        results_map = {}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                results_map[url] = future.result()
            except Exception as exc:
                results_map[url] = {
                    "url": url,
                    "status_code": None,
                    "ok": False,
                    "error": str(exc),
                }

    return [results_map[url] for url in urls if url in results_map]
