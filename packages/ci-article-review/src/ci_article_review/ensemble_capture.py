"""Persist and reload the raw ensemble output so a run can be replayed offline.

The ensemble is the expensive part: at ``maximum`` it is ~29 model calls and
most of a $8 run. Everything after it — consolidation, citation resolution,
report rendering, the history save — operates on data the ensemble already
produced, and until now that data was discarded the moment it was consolidated.
So a one-line change to ``report_markdown`` cost a full ensemble run to exercise.

Across the 25 PRs merged up to 2026-08-15, 15 touched no live-LLM code at all.
Those are exactly the changes this file makes free to iterate on: capture once,
replay as often as you like.

The captured shape is the pipeline's own ``raw_results`` — ``{"model:domain":
result_dict}`` — so a replay hands the pipeline back the identical structure the
dispatch would have produced, and nothing downstream needs to know the
difference.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

#: Bumped when the captured shape changes in a way that older files cannot
#: satisfy. A mismatch refuses to load rather than replaying something subtly
#: wrong — a stale capture that half-works is worse than no capture.
CAPTURE_VERSION = 1


def capture_path_for(report_path):
    """Sibling path for a report: ``run_N_<ts>_report.json`` -> ``..._results.json``."""
    p = Path(report_path)
    return p.with_name(p.name.replace("_report.json", "_results.json"))


def save(path, raw_results, article_title="", run_number=None):
    """Write the raw ensemble output next to the report. Never fails a run."""
    payload = {
        "capture_version": CAPTURE_VERSION,
        "article_title": article_title,
        "run_number": run_number,
        "results": raw_results,
    }
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except (OSError, TypeError, ValueError) as e:
        # Capturing is a convenience for the next iteration, never a reason to
        # lose a run that has already been paid for.
        log.warning("Could not save ensemble capture to %s: %s", path, e)
        return None
    return str(path)


def load(path):
    """Return ``raw_results`` from a capture file.

    Raises ValueError with an actionable message rather than replaying a file
    that does not match what the pipeline now expects.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"No ensemble capture at {p}")
    try:
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not read ensemble capture {p}: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError(f"{p} is not an ensemble capture file")

    version = payload.get("capture_version")
    if version != CAPTURE_VERSION:
        raise ValueError(
            f"{p} was written by capture version {version!r}, this build expects "
            f"{CAPTURE_VERSION}. Re-capture with a live run."
        )

    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{p} contains no ensemble results")

    # Every entry must carry the tags the pipeline re-keys on. A capture missing
    # them would silently collapse into a single "unknown:unknown" result.
    for name, result in results.items():
        if not isinstance(result, dict):
            raise ValueError(f"{p}: entry {name!r} is not a result object")
        if "_model" not in result or "_domain" not in result:
            raise ValueError(
                f"{p}: entry {name!r} is missing _model/_domain — it cannot be "
                "re-keyed. Re-capture with a live run."
            )
    return results


def describe(path):
    """One-line summary of a capture, for logging what a replay is standing on."""
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return str(path)
    results = payload.get("results") or {}
    failed = sum(1 for r in results.values() if isinstance(r, dict) and r.get("failed"))
    title = payload.get("article_title") or "?"
    return (
        f"{Path(path).name}: {len(results)} pass(es), {failed} failed, "
        f"run {payload.get('run_number')}, {title!r}"
    )
