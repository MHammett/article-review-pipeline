"""Wayback Machine availability checker (archive.org CDX API)."""

import logging
import threading
import time
import re
from datetime import datetime, timezone

import requests

from ci_core.http import DEFAULT_HEADERS

from ci_core import redact

log = logging.getLogger(__name__)

_AVAILABILITY_API = "https://archive.org/wayback/available"
_SAVE_API = "https://web.archive.org/save"
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


# archive.org throttles the availability API at IP level with a long window, not
# per-second. Measured 2026-08-12: 12 consecutive requests all returned 429 —
# including the first — and it was still 429 after a 45s cooldown at one request
# every 6 seconds. The pipeline had no pacing and no backoff at all, and the
# result was an archive thread that had been dead for at least two runs: 49
# resolved citations, 0 archived, ~49 rate-limited.
#
# (Re-probed 2026-08-15: six back-to-back lookups all returned 200. The throttle
# is episodic, so the guard has to be always-on rather than tuned to one window.)
#
# Two mechanisms, because one is not enough:
#   * a process-wide minimum interval between calls, serialised on a lock. Claim
#     resolution runs in a thread pool, so without this the pool's width decides
#     the request rate.
#   * retry with backoff that honours Retry-After, and a circuit breaker: once
#     archive.org has said 429 repeatedly, further calls in the same run are
#     skipped rather than spending the run's time collecting more 429s.
#
# Every piece of state below is process-wide, and that is the whole design
# constraint: ``check()`` is called from ``_MAX_PARALLEL`` resolver threads. Two
# consequences that the first version of this code got wrong, both of which
# silently disabled the protection they were meant to provide:
#
#   * **Per-thread backoff is not backoff.** Sleeping inside the failing thread
#     leaves the other workers hammering archive.org at the full pace. A 429 has
#     to move a clock every thread waits on, so the *process* slows down.
#   * **"Consecutive" is not well defined across interleaved threads.** Resetting
#     a shared counter on success let one worker's 200 erase four other workers'
#     refusals, so a run being throttled 4-in-5 would never trip the breaker —
#     precisely the case it exists for. The count is now a per-run budget that
#     only moves up.
_MIN_INTERVAL_SECONDS = 3.0
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 5.0

#: Refused *lookups* tolerated in a run before we stop asking. Counted once per
#: lookup, not once per attempt, so the number means what it says: the
#: ``_MAX_ATTEMPTS`` retries inside one lookup are a single refusal. (Counting
#: attempts made this trip after two lookups rather than five.)
_CIRCUIT_TRIP_AFTER = 5

_pace_lock = threading.Lock()
_last_call_at = 0.0

#: Absolute ``time.monotonic()`` before which no thread may call. A 429 pushes
#: this forward, which is what makes the backoff process-wide.
_blocked_until = 0.0

#: Lookups this run that exhausted every attempt against a 429. Never reset on
#: success — see the note above on why "consecutive" cannot work here.
_rate_limited_lookups = 0


def reset_rate_limit_state():
    """Clear the pacing clock, the backoff, and the circuit breaker.

    Call this at the start of every run. The state is process-wide, so without
    a reset a breaker tripped by one run would skip every archive lookup in the
    next run inside the same process — a silently degraded run whose citations
    all report "skipped" for a limit that expired long ago.
    """
    global _last_call_at, _blocked_until, _rate_limited_lookups
    with _pace_lock:
        _last_call_at = 0.0
        _blocked_until = 0.0
        _rate_limited_lookups = 0


def rate_limited_out():
    """True once the circuit breaker has tripped for this run."""
    with _pace_lock:
        return _rate_limited_lookups >= _CIRCUIT_TRIP_AFTER


def _pace():
    """Block until the shared clock allows another call.

    Waits for whichever is later: ``_MIN_INTERVAL_SECONDS`` since the last call,
    or the end of a backoff that a 429 imposed on every thread. Sleeping while
    holding the lock is deliberate — it is exactly what makes the interval
    process-wide instead of per-thread.
    """
    global _last_call_at
    with _pace_lock:
        now = time.monotonic()
        target = max(_last_call_at + _MIN_INTERVAL_SECONDS, _blocked_until)
        if target > now:
            time.sleep(target - now)
        _last_call_at = time.monotonic()


def _note_rate_limited(retry_after):
    """Back every thread off after a 429, not just the one that hit it."""
    global _blocked_until
    with _pace_lock:
        _blocked_until = max(_blocked_until, time.monotonic() + retry_after)


def _note_lookup_refused():
    """Count one lookup that never got past a 429."""
    global _rate_limited_lookups
    with _pace_lock:
        _rate_limited_lookups += 1


def _retry_after_seconds(resp, attempt):
    """Seconds to wait before retrying, preferring the server's own answer."""
    header = (resp.headers or {}).get("Retry-After") if resp is not None else None
    if header:
        try:
            return min(float(header), 60.0)
        except (TypeError, ValueError):
            pass
    return _BACKOFF_BASE_SECONDS * (2**attempt)


def _get_availability(url, timeout):
    """GET the availability API with pacing, backoff, and a circuit breaker.

    Raises the last exception if every attempt fails, so the caller's existing
    error handling is unchanged.

    There is no local sleep between attempts: a 429 pushes the shared clock out
    and the wait then happens in the next ``_pace()``, which every thread goes
    through. That is the difference between the process backing off and one
    thread backing off while seven others keep the pressure on.
    """
    last_exc = None
    rate_limited = False
    for attempt in range(_MAX_ATTEMPTS):
        _pace()
        try:
            resp = requests.get(
                _AVAILABILITY_API,
                params={"url": url},
                timeout=timeout,
                headers=DEFAULT_HEADERS,
            )
        except Exception as exc:
            last_exc = exc
            continue
        if resp.status_code == 429:
            rate_limited = True
            last_exc = requests.HTTPError(
                f"429 Client Error: Too Many Requests for url: {resp.url}",
                response=resp,
            )
            _note_rate_limited(_retry_after_seconds(resp, attempt))
            continue
        resp.raise_for_status()
        return resp
    # One refused lookup, however many attempts it took to establish that.
    if rate_limited:
        _note_lookup_refused()
    raise last_exc


def check(url, timeout=10, stale_days=None):
    """Check if a URL has a Wayback Machine snapshot and how fresh it is.

    If ``url`` is itself a Wayback Machine snapshot link, it is recognized as
    already-archived: the snapshot date is read from the URL's embedded timestamp
    (no archive-of-an-archive lookup), and ``is_archive_url`` is set. Whether that
    archive link actually resolves is verified separately by the HTTP status check
    in analysis/links.py (the ``ok`` field on the link result).

    Returns a dict with:
      archived        bool | None  — True/False, or None on network error
      is_archive_url  bool         — True when the link itself is a web.archive.org URL
      snapshot_url    str          — direct https://web.archive.org/web/... URL
      snapshot_ts     str          — raw Wayback timestamp (YYYYMMDDHHMMSS)
      snapshot_age_days  int       — days since the snapshot was taken
      snapshot_stale  bool         — True when older than the stale threshold
      error           str          — set only on network/parse failure
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
            "is_archive_url": True,
            "snapshot_url": url,
            "snapshot_ts": ts,
            "snapshot_age_days": age,
            "snapshot_stale": age is not None and age > stale_threshold,
        }

    # Once archive.org has refused repeatedly, stop asking: every further call
    # costs the run several seconds of pacing to collect another 429.
    if rate_limited_out():
        return {
            "url": url,
            "archived": None,
            "error": (
                "skipped: archive.org rate limit tripped earlier this run "
                "(no snapshot lookup attempted)"
            ),
        }
    try:
        resp = _get_availability(url, timeout)
    except Exception as exc:
        log.debug("Wayback availability check failed for %s: %s", url, exc)
        return {"url": url, "archived": None, "error": str(exc)}

    try:
        data = resp.json()
    except ValueError as exc:
        # A 200 that isn't JSON means archive.org served something other than an
        # availability answer — an error or challenge page. Reporting the raw
        # decoder message ("Expecting value: line 1 column 1") sends the reader
        # hunting for a parser bug that isn't there; it is the same misdirection
        # we reported upstream as akamhy/waybackpy#200, where a throttled lookup
        # surfaces as invalid JSON. Say what actually arrived instead.
        log.debug("Wayback availability returned non-JSON for %s: %s", url, exc)
        return {
            "url": url,
            "archived": None,
            "error": (
                f"archive.org returned a non-JSON {resp.status_code} response "
                f"({len(resp.content)} bytes) from the availability API"
            ),
        }

    closest = data.get("archived_snapshots", {}).get("closest", {})
    if not closest.get("available"):
        return {"url": url, "archived": False}

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

    The ``archived is None`` branch is the one that matters. It means the lookup
    never completed — the circuit breaker tripped, or the request failed — which
    is NOT the same as "there is no snapshot". The old wording ("Wayback check
    failed: ...") led with the exception and left a reader to infer the rest;
    since the breaker makes a null the common case rather than a rare one, it
    now says outright that it implies nothing about the page.
    """
    if wb.get("archived") is None:
        return (
            f"NOT CHECKED — the archive.org lookup did not complete "
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
