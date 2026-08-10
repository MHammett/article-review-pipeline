"""
Save and load pipeline run artifacts in pipeline_history/[article_slug]/.
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime, timezone

from .report_markdown import render_report_markdown

log = logging.getLogger(__name__)

# Windows reserved device names that cannot be used as filenames or directory names.
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}

# Saved report filenames: run_<n>_<YYYYmmdd_HHMMSS>_report.json, plus the
# untimestamped run_<n>_report.json shape written by early versions.
_REPORT_NAME_RE = re.compile(r"^run_(?P<run>\d+)_((?P<ts>\d{8}_\d{6})_)?report\.json$")


def _slug(title):
    slug = title.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    slug = slug[:60] or "untitled"
    if slug.lower() in _WINDOWS_RESERVED:
        slug = f"article-{slug}"
    return slug


def _run_dir(history_root, article_title):
    slug = _slug(article_title)
    path = Path(history_root) / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(
    history_root, article_title, run_number, report, corrections_log, run_ts=None
):
    if run_ts is None:
        run_ts = datetime.now(timezone.utc)
    ts_str = run_ts.strftime("%Y%m%d_%H%M%S")

    try:
        d = _run_dir(history_root, article_title)
    except OSError as e:
        log.error(f"Cannot create history directory: {e}")
        return {"report_path": None, "corrections_path": None, "markdown_path": None}

    report_path = d / f"run_{run_number}_{ts_str}_report.json"
    corrections_path = d / f"run_{run_number}_{ts_str}_corrections.log"
    markdown_path = d / f"run_{run_number}_{ts_str}_review.md"

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    except OSError as e:
        log.error(f"Could not write report to {report_path}: {e}")
        return {"report_path": None, "corrections_path": None, "markdown_path": None}

    try:
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(render_report_markdown(report))
    except OSError as e:
        log.warning(f"Could not write markdown review to {markdown_path}: {e}")
        markdown_path = None

    try:
        lines = [
            f"[{c.get('category', '?')}] {c.get('original', '')!r} -> {c.get('replacement', '')!r}  ({c.get('message', '')})"
            for c in corrections_log
        ]
        with open(corrections_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError as e:
        log.warning(f"Could not write corrections log to {corrections_path}: {e}")
        # Report saved successfully — corrections log is supplementary, don't fail the run.
        return {
            "report_path": str(report_path),
            "corrections_path": None,
            "markdown_path": str(markdown_path) if markdown_path else None,
        }

    return {
        "report_path": str(report_path),
        "corrections_path": str(corrections_path),
        "markdown_path": str(markdown_path) if markdown_path else None,
    }


def _report_timestamp(path):
    """Best-effort execution time of a saved report, as an aware UTC datetime.

    ``save_run`` embeds a sortable ``%Y%m%d_%H%M%S`` UTC stamp in the filename,
    which is the cheapest and most direct answer. Reports predating that naming
    are just ``run_N_report.json``, so fall back to the report's own
    ``generated`` field and finally to the file's mtime.
    """
    m = _REPORT_NAME_RE.match(path.name)
    if m and m.group("ts"):
        try:
            return datetime.strptime(m.group("ts"), "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass

    try:
        with open(path, encoding="utf-8") as f:
            generated = json.load(f).get("generated")
    except (OSError, json.JSONDecodeError):
        generated = None
    if generated:
        try:
            ts = datetime.fromisoformat(generated)
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass

    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.min.replace(tzinfo=timezone.utc)


def find_prior_report_path(history_root, article_title, before_ts=None):
    """Path of the most recent execution preceding ``before_ts``, or None.

    Selection is by actual execution time, not by declared run number. The run
    number comes from the handoff document's ``Pipeline run:`` field, which is
    author-declared metadata rather than an execution counter: running the same
    handoff twice writes two ``run_N_*`` reports at the same N. Picking
    ``run_number - 1`` made the second of those compare itself against the
    previous *article version* instead of against the identical run that just
    preceded it, reporting large edits where nothing had been edited.
    """
    try:
        d = _run_dir(history_root, article_title)
    except OSError as e:
        log.warning(f"Cannot read history directory: {e}")
        return None

    candidates = []
    for path in d.glob("run_*_report.json"):
        if not _REPORT_NAME_RE.match(path.name):
            continue
        ts = _report_timestamp(path)
        if before_ts is not None and ts >= before_ts:
            continue
        candidates.append((ts, path.name, path))

    if not candidates:
        return None
    # Filename breaks ties so two reports sharing a timestamp resolve
    # deterministically rather than by directory iteration order.
    return max(candidates)[2]


def load_prior_report(history_root, article_title, before_ts=None):
    """Return ``(report, path)`` for the run preceding ``before_ts``.

    ``(None, None)`` when this article has no earlier report — the first-run
    case — or when the file it found could not be read.
    """
    prior_path = find_prior_report_path(history_root, article_title, before_ts)
    if prior_path is None:
        return None, None
    try:
        with open(prior_path, encoding="utf-8") as f:
            return json.load(f), prior_path
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"Could not load prior report from {prior_path}: {e}")
        return None, None


def append_disposition(history_root, article_title, entry):
    d = _run_dir(history_root, article_title)
    disp_path = d / "disposition.log"
    with open(disp_path, "a", encoding="utf-8") as f:
        timestamp = datetime.now(timezone.utc).isoformat()
        f.write(f"\n[{timestamp}] {entry}\n")
