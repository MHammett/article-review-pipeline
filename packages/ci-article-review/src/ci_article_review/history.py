"""
Save and load pipeline run artifacts in pipeline_history/[article_slug]/.
"""
import json
import re
import logging
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# Windows reserved device names that cannot be used as filenames or directory names.
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
}


def _slug(title):
    slug = title.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    slug = slug[:60] or "untitled"
    if slug.lower() in _WINDOWS_RESERVED:
        slug = f"article-{slug}"
    return slug


def _run_dir(history_root, article_title, run_number):
    slug = _slug(article_title)
    path = Path(history_root) / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(history_root, article_title, run_number, report, corrections_log, run_ts=None):
    if run_ts is None:
        run_ts = datetime.now(timezone.utc)
    ts_str = run_ts.strftime("%Y%m%d_%H%M%S")

    try:
        d = _run_dir(history_root, article_title, run_number)
    except OSError as e:
        log.error(f"Cannot create history directory: {e}")
        return {"report_path": None, "corrections_path": None}

    report_path = d / f"run_{run_number}_{ts_str}_report.json"
    corrections_path = d / f"run_{run_number}_{ts_str}_corrections.log"

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    except OSError as e:
        log.error(f"Could not write report to {report_path}: {e}")
        return {"report_path": None, "corrections_path": None}

    try:
        lines = [
            f"[{c.get('category','?')}] {c.get('original','')!r} -> {c.get('replacement','')!r}  ({c.get('message','')})"
            for c in corrections_log
        ]
        with open(corrections_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError as e:
        log.warning(f"Could not write corrections log to {corrections_path}: {e}")
        # Report saved successfully — corrections log is supplementary, don't fail the run.
        return {"report_path": str(report_path), "corrections_path": None}

    return {"report_path": str(report_path), "corrections_path": str(corrections_path)}


def load_prior_report(history_root, article_title, run_number):
    if run_number <= 1:
        return None
    d = _run_dir(history_root, article_title, run_number)
    matches = sorted(d.glob(f"run_{run_number - 1}_*_report.json"))
    if not matches:
        return None
    prior_path = matches[-1]
    try:
        with open(prior_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"Could not load prior report from {prior_path}: {e}")
        return None


def append_disposition(history_root, article_title, entry):
    d = _run_dir(history_root, article_title, 1)
    disp_path = d / "disposition.log"
    with open(disp_path, "a", encoding="utf-8") as f:
        timestamp = datetime.now(timezone.utc).isoformat()
        f.write(f"\n[{timestamp}] {entry}\n")
