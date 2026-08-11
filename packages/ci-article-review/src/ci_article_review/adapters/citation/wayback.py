"""Wayback Machine availability checker (archive.org CDX API)."""

import logging
import re
import threading
import time
from datetime import datetime, timezone

import requests

from ci_core.http import DEFAULT_HEADERS

from ci_core import redact

log = logging.getLogger(__name__)

_AVAILABILITY_API = "https://archive.org/wayback/available"
_SAVE_API = "https://web.archive.org/save"

#: Bound on concurrent availability lookups across the whole process.
#:
#: Two independent pools call ``check`` — citation resolution (8 workers) and
#: link analysis (10) — and neither knows about the other. archive.org sees one
#: client, and in a real run every one of 65 lookups came back HTTP 429. The
#: result was a report where every citation read ``archived: None``, which is
#: "we never found out", displayed where a reader looks for "is this archived".
#:
#: The semaphore is module-level for the same reason the problem is: the limit
#: belongs to archive.org's view of us, not to any one caller's pool.
_MAX_PARALLEL_CHECKS = 3
_CHECK_SEMAPHORE = threading.Semaphore(_MAX_PARALLEL_CHECKS)

#: Retries for a rate-limited availability lookup, and the base for exponential
#: backoff between them. A ``Retry-After`` header wins over the computed delay.
#: Kept small: this is a nice-to-have enrichment on a citation, and a run should
#: not spend minutes waiting on it.
_CHECK_RETRIES = 3
_CHECK_BACKOFF_SECONDS = 2.0

#: Longest we will honour a ``Retry-After`` for. archive.org has been known to
#: send minutes; waiting that long per URL would stall the run for something
#: that is not load-bearing.
_MAX_RETRY_AFTER_SECONDS = 10.0
_STALE_DAYS = (
    180  # default — overridden by pipeline.wayback_snapshot_stale_days in user.yaml
)

# Matches a Wayback Machine snapshot URL and captures its embedded timestamp:
#   https://web.archive.org/web/20250101000000/https://example.com/...
_ARCHIVE_URL_RE = re.compile(
    r"https?://web\.archive\.org/web/(\d{4,14})(?:[a-z_]*)?/", re.IGNORECASE
)

#: HTTP statuses where the origin is reachable but refuses to serve *us* the
#: document. The resource itself is not claimed to be gone, so reading
#: archive.org's copy answers the question the origin declined to.
#:
#: Deliberately excluded:
#:   404 / 410 — the resource is genuinely gone. Surfacing that is the whole
#:     point of link validation; an archive copy would mask a real problem the
#:     author has to fix by re-sourcing the claim.
#:   5xx — the origin's own failure, not a refusal aimed at us. A transient 5xx
#:     will be fine by the time a reader clicks, and a persistent one means the
#:     source needs replacing; standing in an archived copy hides both. (A 5xx
#:     is also the shape a misconfigured origin returns for a page it no longer
#:     has, so treating it as "reachable content" is not safe.)
_FALLBACK_STATUSES = {
    401: "auth_required",
    403: "blocked",
    # Rate limiting is transient in a way the others aren't, but it is still
    # "the origin won't serve us right now" rather than "this is gone", and a
    # run shouldn't report a good link as broken because we asked too fast.
    # The distinct reason label keeps that visible in the report.
    429: "rate_limited",
}

#: Human-readable phrasing for each reason, for report output.
FALLBACK_REASON_LABELS = {
    "auth_required": "401 auth required",
    "blocked": "403 blocked",
    "rate_limited": "429 rate limited",
    "timeout": "origin timed out",
    "unreachable": "origin unreachable",
}


def fallback_reason_for_status(status):
    """Reason label if an HTTP ``status`` warrants an archive fallback, else None.

    See ``_FALLBACK_STATUSES`` for what is in scope and, more importantly, what
    is deliberately not.
    """
    return _FALLBACK_STATUSES.get(status)


def fallback_reason_for_exception(exc):
    """Reason label if a fetch exception warrants an archive fallback, else None.

    Covers the "we never reached the origin" failures — connect/read timeouts
    and connection errors (which is where ``requests`` puts DNS resolution
    failures). These say nothing about whether the resource exists, only that
    we couldn't ask, so an archived copy is a legitimate substitute in exactly
    the way it is for a 403.

    An ``HTTPError`` is dispatched to ``fallback_reason_for_status`` so callers
    with one bare ``except`` don't have to special-case it.
    """
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        return fallback_reason_for_status(status)
    # Timeout first: ConnectTimeout subclasses both Timeout and ConnectionError,
    # and "timed out" is the more specific description of it.
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "unreachable"
    return None


def _age_days_from_timestamp(ts):
    """Days since a Wayback timestamp (YYYYMMDD...), or None if unparseable."""
    if len(ts) >= 8:
        try:
            snap_dt = datetime.strptime(ts[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - snap_dt).days
        except ValueError:
            pass
    return None


def _retry_after_seconds(resp):
    """Seconds to wait per the response's ``Retry-After``, if it has a usable one."""
    raw = (resp.headers or {}).get("Retry-After") if resp is not None else None
    if not raw:
        return None
    try:
        return min(float(str(raw).strip()), _MAX_RETRY_AFTER_SECONDS)
    except (TypeError, ValueError):
        # The HTTP-date form is legal and archive.org does not use it. Fall back
        # to computed backoff rather than parsing a format we never see.
        return None


def _get_availability(url, timeout):
    """Fetch the availability API, retrying while archive.org rate-limits us.

    Only 429 is retried. A 5xx is archive.org's own problem and retrying inside
    a run rarely helps; anything else is not transient.
    """
    last_exc = None
    for attempt in range(_CHECK_RETRIES):
        with _CHECK_SEMAPHORE:
            resp = requests.get(
                _AVAILABILITY_API,
                params={"url": url},
                timeout=timeout,
                headers=DEFAULT_HEADERS,
            )
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp
            last_exc = requests.exceptions.HTTPError(
                f"429 Client Error: Too Many Requests for url: {_AVAILABILITY_API}",
                response=resp,
            )
        # Sleep outside the semaphore so a waiting thread can take the slot.
        if attempt < _CHECK_RETRIES - 1:
            delay = _retry_after_seconds(resp)
            if delay is None:
                delay = _CHECK_BACKOFF_SECONDS * (2**attempt)
            log.debug(
                "Wayback rate-limited for %s; retrying in %.1fs (attempt %d/%d)",
                url,
                delay,
                attempt + 1,
                _CHECK_RETRIES,
            )
            time.sleep(delay)
    raise last_exc


def check(url, timeout=10, stale_days=None):
    """Check if a URL has a Wayback Machine snapshot and how fresh it is.

    If ``url`` is itself a Wayback Machine snapshot link, it is recognized as
    already-archived: the snapshot date is read from the URL's embedded timestamp
    (no archive-of-an-archive lookup), and ``is_archive_url`` is set. Whether that
    archive link actually resolves is verified separately by the HTTP status check
    in analysis/links.py (the ``ok`` field on the link result).

    Rate-limited lookups are retried with backoff and, across the process, held
    to ``_MAX_PARALLEL_CHECKS`` at a time. When the lookup still does not
    complete, the result says so: ``archived`` is None and ``checked`` is False.
    Those two are not the same claim as ``archived: False`` and must never be
    read as one — see ``format_summary``.

    Returns a dict with:
      archived        bool | None  — True/False, or None when the check couldn't run
      checked         bool         — False when archive.org was never successfully asked
      is_archive_url  bool         — True when the link itself is a web.archive.org URL
      snapshot_url    str          — direct https://web.archive.org/web/... URL
      snapshot_ts     str          — raw Wayback timestamp (YYYYMMDDHHMMSS)
      snapshot_age_days  int       — days since the snapshot was taken
      snapshot_stale  bool         — True when older than the stale threshold
      error           str          — set only on network/parse failure
      rate_limited    bool         — set when the failure was archive.org throttling us
    """
    stale_threshold = stale_days if stale_days is not None else _STALE_DAYS

    # The link is already a Wayback snapshot — read its date from the URL itself.
    m = _ARCHIVE_URL_RE.search(url)
    if m:
        ts = m.group(1)
        age = _age_days_from_timestamp(ts)
        return {
            "url": url,
            "archived": True,
            "checked": True,
            "is_archive_url": True,
            "snapshot_url": url,
            "snapshot_ts": ts,
            "snapshot_age_days": age,
            "snapshot_stale": age is not None and age > stale_threshold,
        }

    try:
        resp = _get_availability(url, timeout)
        data = resp.json()
    except Exception as exc:
        log.debug("Wayback availability check failed for %s: %s", url, exc)
        result = {
            "url": url,
            "archived": None,
            "checked": False,
            "error": str(exc),
        }
        if isinstance(exc, requests.exceptions.HTTPError) and "429" in str(exc):
            result["rate_limited"] = True
        return result

    closest = data.get("archived_snapshots", {}).get("closest", {})
    if not closest.get("available"):
        return {"url": url, "archived": False, "checked": True}

    ts = closest.get("timestamp", "")
    snapshot_url = closest.get("url", "")
    snapshot_age_days = None

    if len(ts) >= 8:
        try:
            snap_dt = datetime.strptime(ts[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
            snapshot_age_days = (datetime.now(timezone.utc) - snap_dt).days
        except ValueError:
            pass

    stale_threshold = stale_days if stale_days is not None else _STALE_DAYS
    return {
        "url": url,
        "archived": True,
        "checked": True,
        "snapshot_url": snapshot_url,
        "snapshot_ts": ts,
        "snapshot_age_days": snapshot_age_days,
        "snapshot_stale": snapshot_age_days is not None
        and snapshot_age_days > stale_threshold,
    }


def submit(url, timeout=30, access_key=None, secret_key=None):
    """Request that archive.org capture and archive ``url`` (Save Page Now / SPN2).

    Fire-and-forget by design: SPN2 captures run asynchronously on archive.org's
    side and a real page capture can take anywhere from seconds to minutes. This
    does not poll the job-status endpoint for completion — the pipeline should
    not block on someone else's crawl. A later run's ``check()`` call will
    naturally see the new snapshot once archive.org finishes it.

    With ``access_key``/``secret_key`` (an archive.org S3-style API key pair,
    from https://archive.org/account/s3.php), submission goes through the
    authenticated SPN2 endpoint, which gets higher rate limits and returns a
    job id. Without credentials, falls back to the unauthenticated capture
    trigger (``GET /save/<url>``), which works but is subject to tighter,
    unpredictable archive.org rate limits.

    Returns a dict with:
      url        str
      submitted  bool  — True if archive.org accepted the request
      job_id     str | None — SPN2 job id, only set for authenticated submission
      error      str  — set only on failure
    """
    headers = dict(DEFAULT_HEADERS)
    try:
        if access_key and secret_key:
            headers["Authorization"] = f"LOW {access_key}:{secret_key}"
            resp = requests.post(
                _SAVE_API,
                data={"url": url},
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            return {"url": url, "submitted": True, "job_id": payload.get("job_id")}

        resp = requests.get(
            f"{_SAVE_API}/{url}",
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return {"url": url, "submitted": True, "job_id": None}
    except Exception as exc:
        err = redact.redact_value(redact.redact_url_keys(str(exc)), secret_key)
        log.warning("Wayback submission failed for %s: %s", url, err)
        return {"url": url, "submitted": False, "error": err}


def format_summary(wb):
    """One-line human-readable summary of a wayback result.

    The first branch is the point of this function. "We never asked" and "we
    asked and there is no snapshot" are different facts, and a reader scanning
    a citation list will act on them differently — the second is a reason to go
    archive the page, the first is not a finding at all.
    """
    if wb.get("archived") is None:
        if wb.get("rate_limited"):
            return (
                "NOT CHECKED — archive.org rate-limited this run (HTTP 429). "
                "This says nothing about whether the page is archived."
            )
        return (
            f"NOT CHECKED — the archive.org lookup failed "
            f"({wb.get('error', 'unknown error')}). This says nothing about "
            f"whether the page is archived."
        )
    if not wb.get("archived"):
        return "Not archived in Wayback Machine"
    age = wb.get("snapshot_age_days")
    stale = wb.get("snapshot_stale")
    age_str = f"{age}d ago" if age is not None else "age unknown"
    flag = " [STALE]" if stale else ""
    return f"Archived — latest snapshot {age_str}{flag}: {wb.get('snapshot_url', '')}"
