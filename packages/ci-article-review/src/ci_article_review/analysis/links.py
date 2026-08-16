"""URL extraction, HTTP status validation, and Wayback Machine archive checks."""

import concurrent.futures
import logging
import re

import requests

from ci_core.http import DEFAULT_HEADERS, impersonating_get, safe_get
from ci_core.http import is_public_host as _core_is_public_host

from ..adapters.citation.wayback import (
    check as wayback_check,
    fallback_reason_for_exception,
    fallback_reason_for_status,
)

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
    """Link validation's view of the SSRF guard: fail *open* on DNS failure.

    Delegates to ``ci_core.http.is_public_host``. The guard moved to ci-core so
    adapters/citation/ could reach it too — it previously lived here, where
    ``adapters -> analysis`` would have closed an import cycle, which is why
    model-supplied URLs went unvalidated (audit finding 2).

    This wrapper keeps the fail-open choice, which is right *here* and wrong
    elsewhere: the job of link validation is to tell the author why a link is
    bad, so an unresolvable host should produce the real DNS error from the HTTP
    layer rather than a security refusal. Anything that fetches a body and then
    feeds it to a model uses the fail-closed default instead.
    """
    return _core_is_public_host(url, fail_open_on_dns_error=True)


def _wayback_fallback(url, timeout):
    """When the origin won't (or can't) serve us the page, try archive.org's
    snapshot instead of giving up.

    archive.org serves its own cached copy, so a site blocking our fetch —
    or a host we never reached at all — doesn't block the archived one. Which
    failures qualify is decided by ``fallback_reason_for_status`` /
    ``fallback_reason_for_exception`` in the wayback module; a 404 (genuinely
    gone) and a 5xx (the origin's own problem) deliberately do not. A single
    attempt, no retry loop.

    Returns the snapshot URL on success, or None if there's no snapshot or
    the snapshot itself doesn't resolve.
    """
    wb = wayback_check(url, timeout=timeout)
    snapshot_url = wb.get("snapshot_url")
    # Covers both False ("no snapshot exists") and None ("the lookup never
    # completed") on purpose: with no snapshot URL there is nothing to fetch
    # either way. The link's own rendering is what has to keep them apart — see
    # the `archive not checked` branch in pipeline's link summary — because a
    # link unrecoverable due to the circuit breaker is not the same finding as
    # one with genuinely no archive.
    if not wb.get("archived") or not snapshot_url:
        return None
    try:
        snap_resp = safe_get(snapshot_url, timeout=timeout)
    except Exception:
        return None
    if snap_resp.status_code >= 400:
        return None
    return snapshot_url


def _finalize_http_result(url, resp, timeout):
    final_url = resp.url
    result = {
        "status_code": resp.status_code,
        "ok": resp.status_code < 400,
        "redirected_to": final_url if final_url != url else None,
        "verified_via": "direct",
    }
    reason = fallback_reason_for_status(resp.status_code)
    if reason:
        _apply_wayback_fallback(result, url, reason, timeout)
    return result


def _apply_wayback_fallback(result, url, reason, timeout):
    """Try the archive fallback and, if it lands, mark ``result`` as recovered.

    ``origin_failure`` records *why* the origin didn't serve us the page and is
    set either way — a link read from the archive after a timeout stays
    distinguishable from one read after a 403, and a link that stayed broken
    is still distinguishable from a confirmed 404. The origin's own
    status/error is left on the result untouched for the same reason: a
    recovered link must never look like a clean direct fetch.
    """
    result["origin_failure"] = reason
    snapshot_url = _wayback_fallback(url, timeout)
    if not snapshot_url:
        return False
    result["ok"] = True
    result["verified_via"] = "wayback_fallback"
    result["wayback_snapshot_url"] = snapshot_url
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
            headers=DEFAULT_HEADERS,
        )
        # 405 means HEAD is not allowed. 403 often means the same thing in
        # practice: some servers and WAFs reject HEAD specifically while serving
        # GET to the identical client. Retrying with GET is a method change, not
        # an identity change — the honest User-Agent is unchanged, so a site with
        # a deliberate bot policy still gets to enforce it (see ci_core.http).
        #
        # Measured on six real 403s from the 2026-08-12 run: one was served
        # normally on GET, and one turned out to be a genuine 404 that HEAD had
        # been reporting as 403. That second case is why this matters most —
        # a dead link was being excused as "403 blocked, likely still valid".
        if resp.status_code in (403, 405):
            with requests.get(
                url,
                allow_redirects=True,
                timeout=timeout,
                headers=DEFAULT_HEADERS,
                stream=True,
            ) as get_resp:
                # Last tier: a browser TLS fingerprint. Only for a link still
                # refusing an honest request, and only to confirm a source the
                # article already cites is real — an unverifiable citation is a
                # worse outcome than a spoofed handshake. Returns None when
                # curl_cffi is absent or the block holds, in which case the
                # original result stands and the Wayback fallback runs as before.
                if get_resp.status_code == 403:
                    impersonated = impersonating_get(url, timeout=timeout)
                    if impersonated is not None:
                        return {
                            "status_code": impersonated.status_code,
                            "ok": True,
                            "error": None,
                            "verified_via": "tls_impersonation",
                            "final_url": str(getattr(impersonated, "url", url)),
                        }
                return _finalize_http_result(url, get_resp, timeout)
        return _finalize_http_result(url, resp, timeout)
    except Exception as exc:
        return _finalize_error_result(url, exc, timeout)


def _finalize_error_result(url, exc, timeout):
    """Build the result for a fetch that never produced a response.

    A timeout or a DNS/connection failure means we couldn't reach the origin —
    which says nothing about whether the page exists — so these get the same
    archive fallback a 403 does. The error stays on the result either way: a
    recovered link is reported as read-from-archive, never as a clean fetch.
    """
    error = "timeout" if isinstance(exc, requests.exceptions.Timeout) else str(exc)
    result = {"status_code": None, "ok": False, "error": error}
    reason = fallback_reason_for_exception(exc)
    if reason:
        _apply_wayback_fallback(result, url, reason, timeout)
    return result


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
      error         str   — set only on network error; still set when the link
                            was recovered from the archive, since the origin
                            really did fail
      verified_via  str   — "direct", or "wayback_fallback" when the live URL
                            couldn't be read but an archive.org snapshot could
      origin_failure str  — why the origin didn't serve us the page, whether or
                            not the archive fallback then succeeded: "blocked"
                            (403), "auth_required" (401), "rate_limited" (429),
                            "timeout", or "unreachable" (DNS/connection error).
                            Absent for a 404/410/5xx, which get no fallback.
      wayback_snapshot_url str — the snapshot fetched, set only when verified_via
                            is "wayback_fallback"
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
