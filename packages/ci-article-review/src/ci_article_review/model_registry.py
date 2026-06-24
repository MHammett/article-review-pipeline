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

import yaml as _yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallbacks — used only when configs/model_registry.yaml is missing.
# ---------------------------------------------------------------------------

_SUPERSEDED_FALLBACK: dict[str, dict] = {
    "gpt-4":               {"replacement": "gpt-5.4",      "note": "GPT-5 family available (2026)"},
    "gpt-4-turbo":         {"replacement": "gpt-5.4",      "note": "GPT-5 family available (2026)"},
    "gpt-4-turbo-preview": {"replacement": "gpt-5.4"},
    "gpt-4o":              {"replacement": "gpt-5.4",      "note": "GPT-5 family available (2026)"},
    "gpt-4o-mini":         {"replacement": "gpt-5.4-mini", "note": "GPT-5 family available (2026)"},
    "gpt-3.5-turbo":       {"replacement": "gpt-5.4-mini"},
    "o3":                  {"replacement": "gpt-5.5",      "note": "gpt-5.5 with reasoning_effort: xhigh supersedes o3 at maximum preset"},
    "o4-mini":             {"replacement": "gpt-5.4",      "note": "gpt-5.4 with reasoning_effort supersedes o4-mini at balanced/thorough presets"},
    "gemini-pro":                    {"replacement": "gemini-2.5-flash"},
    "gemini-1.0-pro":                {"replacement": "gemini-2.5-flash"},
    "gemini-1.5-flash":              {"replacement": "gemini-2.5-flash"},
    "gemini-1.5-flash-latest":       {"replacement": "gemini-2.5-flash"},
    "gemini-1.5-pro":                {"replacement": "gemini-2.5-pro"},
    "gemini-1.5-pro-latest":         {"replacement": "gemini-2.5-pro"},
    "gemini-2.0-flash":              {"replacement": "gemini-2.5-flash"},
    "gemini-2.0-flash-exp":          {"replacement": "gemini-2.5-flash"},
    "gemini-2.0-flash-thinking-exp": {"replacement": "gemini-2.5-flash"},
    "grok-beta":          {"replacement": "grok-4.3",     "note": "Grok 4.x family available (2026)"},
    "grok-2-latest":      {"replacement": "grok-4.3"},
    "grok-3-latest":      {"replacement": "grok-4.3",     "note": "Grok 4.x family available (2026)"},
    "grok-3-mini-latest": {"replacement": "grok-build-0.1"},
    "claude-3-haiku-20240307":  {"replacement": "claude-haiku-4-5-20251001"},
    "claude-3-sonnet-20240229": {"replacement": "claude-sonnet-4-6"},
    "claude-3-opus-20240229":   {"replacement": "claude-opus-4-8"},
    "claude-haiku-3-5":         {"replacement": "claude-haiku-4-5-20251001"},
    "claude-sonnet-3-5":        {"replacement": "claude-sonnet-4-6"},
    "claude-sonnet-3-7":        {"replacement": "claude-sonnet-4-6"},
    "claude-opus-4-5":          {"replacement": "claude-opus-4-8"},
    "claude-opus-4-7":          {"replacement": "claude-opus-4-8"},
    "pplx-7b-online":     {"replacement": "sonar"},
    "pplx-70b-online":    {"replacement": "sonar-pro"},
    "sonar-small-online": {"replacement": "sonar"},
    "sonar-medium-online":{"replacement": "sonar-pro"},
    "mistral-small":             {"replacement": "mistral-small-latest"},
    "mistral-medium":            {"replacement": "mistral-large-latest", "note": "mistral-medium tier retired"},
    "mistral-7b-instruct":       {"replacement": "mistral-small-latest"},
    "mixtral-8x7b-instruct":     {"replacement": "mistral-large-latest"},
    "mixtral-8x22b-instruct":    {"replacement": "mistral-large-latest"},
    "magistral-medium-latest":   {"replacement": "mistral-medium-3-5", "note": "deprecated 5/22/2026, retires 7/31/2026"},
    "magistral-medium-2509":     {"replacement": "mistral-medium-3-5", "note": "deprecated 5/22/2026, retires 7/31/2026"},
    "magistral-small-latest":    {"replacement": "mistral-small-latest", "note": "deprecated 4/30/2026, retires 7/31/2026"},
    "magistral-small-2509":      {"replacement": "mistral-small-latest", "note": "deprecated 4/30/2026, retires 7/31/2026"},
}

_NEWER_AVAILABLE_FALLBACK: dict[str, dict] = {
    "gpt-5.4": {
        "newer": "gpt-5.5",
        "note": "gpt-5.5 ($5/$30 MTok) available; gpt-5.4 still best value for most use cases",
    },
    "claude-sonnet-4-6": {
        "newer": "claude-opus-4-8",
        "note": "claude-opus-4-8 offers adaptive thinking (always on); sonnet-4-6 remains better value at $3/$15 MTok",
    },
}

_REGISTRY_DATE_FALLBACK = datetime.date(2026, 6, 22)
_STALE_NOTICE_DAYS_FALLBACK  = 60
_STALE_WARNING_DAYS_FALLBACK = 120

# ---------------------------------------------------------------------------
# Load from configs/model_registry.yaml
# ---------------------------------------------------------------------------

def _load_registry():
    yaml_path = Path(__file__).parent / "configs" / "model_registry.yaml"
    if not yaml_path.exists():
        return (
            _SUPERSEDED_FALLBACK,
            _NEWER_AVAILABLE_FALLBACK,
            _REGISTRY_DATE_FALLBACK,
            _STALE_NOTICE_DAYS_FALLBACK,
            _STALE_WARNING_DAYS_FALLBACK,
        )
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}

        def _normalise(raw):
            """Convert YAML dict-of-dicts to {model_id: {replacement, note}} form."""
            result = {}
            for model_id, info in (raw or {}).items():
                if isinstance(info, dict):
                    result[str(model_id)] = {k: v for k, v in info.items()}
            return result

        superseded       = _normalise(data.get("superseded"))
        newer_available  = _normalise(data.get("newer_available"))
        reg_date_raw     = data.get("registry_date", _REGISTRY_DATE_FALLBACK.isoformat())
        reg_date         = datetime.date.fromisoformat(str(reg_date_raw))
        notice_days      = int(data.get("stale_notice_days",  _STALE_NOTICE_DAYS_FALLBACK))
        warning_days     = int(data.get("stale_warning_days", _STALE_WARNING_DAYS_FALLBACK))
        return superseded, newer_available, reg_date, notice_days, warning_days
    except Exception as exc:
        log.warning("Could not load configs/model_registry.yaml (%s) — using built-in defaults", exc)
        return (
            _SUPERSEDED_FALLBACK,
            _NEWER_AVAILABLE_FALLBACK,
            _REGISTRY_DATE_FALLBACK,
            _STALE_NOTICE_DAYS_FALLBACK,
            _STALE_WARNING_DAYS_FALLBACK,
        )


_SUPERSEDED, _NEWER_AVAILABLE, REGISTRY_DATE, _STALE_NOTICE_DAYS, _STALE_WARNING_DAYS = _load_registry()


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
        enabled  = cfg.get("enabled", True) if isinstance(cfg, dict) else True

        if not enabled or not model_id:
            continue

        if model_id in _SUPERSEDED:
            info = _SUPERSEDED[model_id]
            warnings.append({
                "provider": provider,
                "model":       model_id,
                "replacement": info["replacement"],
                "note":        info.get("note", ""),
            })
        elif model_id in _NEWER_AVAILABLE:
            info = _NEWER_AVAILABLE[model_id]
            notices.append({
                "provider": provider,
                "model":  model_id,
                "newer":  info["newer"],
                "note":   info.get("note", ""),
            })

    today = datetime.date.today()
    age = (today - REGISTRY_DATE).days

    return {
        "warnings":           warnings,
        "notices":            notices,
        "registry_date":      REGISTRY_DATE.isoformat(),
        "registry_age_days":  age,
        "registry_stale":     age >= _STALE_NOTICE_DAYS,
        "registry_warning":   age >= _STALE_WARNING_DAYS,
    }
