"""Model currency registry.

Tracks which model IDs are current, which have been superseded, and when this
registry was last verified against live provider documentation.  Called once
per pipeline run to surface model-staleness warnings in the report.

Registry data is loaded from configs/model_registry.yaml at import time so
model deprecation entries and the registry date can be updated without editing
Python.  The hardcoded dicts below are used only when the YAML file is missing.

To update the registry:
  1. Edit configs/model_registry.yaml — add/remove entries, bump registry_date.
  2. No code change needed.
"""

import datetime
import logging
from pathlib import Path

import yaml as _yaml  # noqa: F401

from ci_core.config_helpers import PackagedConfigError, load_packaged_yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallbacks — used only when configs/model_registry.yaml is missing.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Load from configs/model_registry.yaml
# ---------------------------------------------------------------------------


def _load_registry():
    """Load the model registry from the packaged configs/model_registry.yaml.

    Raises PackagedConfigError rather than falling back to duplicate tables in
    Python (audit finding 14) — see ci_core.config_helpers.load_packaged_yaml.
    Model names and deprecation dates move constantly, which is precisely why
    keeping a second copy in lockstep was the wrong trade.
    """
    yaml_path = Path(__file__).parent.parent / "configs" / "model_registry.yaml"
    data = load_packaged_yaml(yaml_path)

    def _normalise(raw, label):
        """Convert YAML dict-of-dicts to {model_id: {replacement, note}} form."""
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise PackagedConfigError(f"{yaml_path}: {label!r} must be a mapping")
        return {
            str(model_id): dict(info)
            for model_id, info in raw.items()
            if isinstance(info, dict)
        }

    superseded = _normalise(data.get("superseded"), "superseded")
    newer_available = _normalise(data.get("newer_available"), "newer_available")

    reg_date_raw = data.get("registry_date")
    if not reg_date_raw:
        raise PackagedConfigError(f"{yaml_path}: missing 'registry_date'")
    try:
        reg_date = datetime.date.fromisoformat(str(reg_date_raw))
    except ValueError as exc:
        raise PackagedConfigError(
            f"{yaml_path}: registry_date {reg_date_raw!r} is not an ISO date"
        ) from exc

    notice_days = int(data.get("stale_notice_days", 60))
    warning_days = int(data.get("stale_warning_days", 120))
    if notice_days > warning_days:
        raise PackagedConfigError(
            f"{yaml_path}: stale_notice_days ({notice_days}) must not exceed "
            f"stale_warning_days ({warning_days})"
        )
    return superseded, newer_available, reg_date, notice_days, warning_days


(
    _SUPERSEDED,
    _NEWER_AVAILABLE,
    REGISTRY_DATE,
    _STALE_NOTICE_DAYS,
    _STALE_WARNING_DAYS,
) = _load_registry()

# Public aliases for the loaded registry tables. These cross a package boundary
# (ci-article-review's `ci-discover` reports on them), so they carry public
# names — see docs/NAMING.md, "Dependency direction". The underscored names
# above are kept because the parity tests and this module's own internals refer
# to them alongside their _FALLBACK counterparts.
SUPERSEDED = _SUPERSEDED
NEWER_AVAILABLE = _NEWER_AVAILABLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_model_currency(model_configs: dict) -> dict:
    """Inspect model_configs for outdated or superseded model IDs.

    ``model_configs`` is the normalized models dict from ``merge_configs()``
    (each value is already a dict with at least ``model`` and ``provider`` keys).

    Returns::

        {
          "warnings": [
              {"provider": str, "model": str, "replacement": str, "note": str},
              ...
          ],
          "notices": [
              {"provider": str, "model": str, "newer": str, "note": str},
              ...
          ],
          "registry_date": "YYYY-MM-DD",
          "registry_age_days": int,
          "registry_stale": bool,      # True at 60+ days
          "registry_warning": bool,    # True at 120+ days
        }
    """
    warnings = []
    notices = []

    for provider, cfg in (model_configs or {}).items():
        model_id = cfg.get("model", "") if isinstance(cfg, dict) else str(cfg)
        enabled = cfg.get("enabled", True) if isinstance(cfg, dict) else True

        if not enabled or not model_id:
            continue

        if model_id in _SUPERSEDED:
            info = _SUPERSEDED[model_id]
            warnings.append(
                {
                    "provider": provider,
                    "model": model_id,
                    "replacement": info["replacement"],
                    "note": info.get("note", ""),
                }
            )
        elif model_id in _NEWER_AVAILABLE:
            info = _NEWER_AVAILABLE[model_id]
            notices.append(
                {
                    "provider": provider,
                    "model": model_id,
                    "newer": info["newer"],
                    "note": info.get("note", ""),
                }
            )

    today = datetime.date.today()
    age = (today - REGISTRY_DATE).days

    return {
        "warnings": warnings,
        "notices": notices,
        "registry_date": REGISTRY_DATE.isoformat(),
        "registry_age_days": age,
        "registry_stale": age >= _STALE_NOTICE_DAYS,
        "registry_warning": age >= _STALE_WARNING_DAYS,
    }
