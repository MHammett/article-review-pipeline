"""
Save and load pipeline run artifacts in pipeline_history/[article_slug]/.
"""
import json
import re
import os
from pathlib import Path
from datetime import datetime, timezone


def _slug(title):
    slug = title.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:60] or "untitled"


def _run_dir(history_root, article_title, run_number):
    slug = _slug(article_title)
    path = Path(history_root) / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(history_root, article_title, run_number, report, corrections_log):
    d = _run_dir(history_root, article_title, run_number)
    report_path = d / f"run_{run_number}_report.json"
    corrections_path = d / f"run_{run_number}_corrections.log"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    lines = []
    for c in corrections_log:
        lines.append(
            f"[{c.get('category','?')}] {c.get('original','')!r} -> {c.get('replacement','')!r}  ({c.get('message','')})"
        )
    with open(corrections_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"report_path": str(report_path), "corrections_path": str(corrections_path)}


def load_prior_report(history_root, article_title, run_number):
    """Load the report from run_number - 1, if it exists."""
    if run_number <= 1:
        return None
    d = _run_dir(history_root, article_title, run_number)
    prior_path = d / f"run_{run_number - 1}_report.json"
    if not prior_path.exists():
        return None
    with open(prior_path, encoding="utf-8") as f:
        return json.load(f)


def append_disposition(history_root, article_title, entry):
    d = _run_dir(history_root, article_title, 1)
    disp_path = d / "disposition.log"
    with open(disp_path, "a", encoding="utf-8") as f:
        timestamp = datetime.now(timezone.utc).isoformat()
        f.write(f"\n[{timestamp}] {entry}\n")
