"""Wayback Machine availability checker (archive.org CDX API)."""

import logging
import re
from datetime import datetime, timezone

import requests

from ci_core.http import DEFAULT_HEADERS

from ... import redact

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


def _age_days_from_timestamp(ts):
    """Days since a Wayback timestamp (YYYYMMDD...), or None if unparseable."""
    if len(ts) >= 8:
        try:
            snap_dt = datetime.strptime(ts[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - snap_dt).days
        except ValueError:
            pass
    return None


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

    try:
        resp = requests.get(
            _AVAILABILITY_API,
            params={"url": url},
            timeout=timeout,
            headers=DEFAULT_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.debug("Wayback availability check failed for %s: %s", url, exc)
        return {"url": url, "archived": None, "error": str(exc)}

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
    """One-line human-readable summary of a wayback result."""
    if wb.get("archived") is None:
        return f"Wayback check failed: {wb.get('error', 'unknown error')}"
    if not wb.get("archived"):
        return "Not archived in Wayback Machine"
    age = wb.get("snapshot_age_days")
    stale = wb.get("snapshot_stale")
    age_str = f"{age}d ago" if age is not None else "age unknown"
    flag = " [STALE]" if stale else ""
    return f"Archived — latest snapshot {age_str}{flag}: {wb.get('snapshot_url', '')}"
