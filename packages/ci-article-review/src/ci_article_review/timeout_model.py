"""Sliding-scale timeout model.

Computes a per-model timeout from the draft size, the model, and the reasoning
effort, so timeouts track real call latency instead of being hand-tuned per preset.
Calibrated from runs on 2026-06-22; see configs/timeouts.yaml.

Since the review adapters stream (SSE), the value computed here is the per-task
thread WALL-CLOCK BACKSTOP (enforced by pipeline._run_with_timeout), not the HTTP
socket timeout. The socket timeout is a small constant inter-token read gap held by
the adapters (adapters/review/streaming.py). This module is unchanged by streaming —
only the *meaning* of its output shifted from "socket read timeout" to "wall-clock
backstop"; both are sized the same way (total generation time still bounds it).

    effective = clamp( base × size_mult × model_mult × effort_mult, floor, ceiling )

Config is loaded from configs/timeouts.yaml at import time; the hardcoded
_FALLBACK is used only when the YAML is missing or unreadable.
"""

import logging
from pathlib import Path

import yaml as _yaml

log = logging.getLogger(__name__)

# Hardcoded fallback — kept in parity with configs/timeouts.yaml (see tests).
_FALLBACK = {
    "base_seconds": 60,
    "floor_seconds": 60,
    "variance_margin": 1.25,
    "size_multipliers": [
        {"max_chars": 5000,   "mult": 0.4},
        {"max_chars": 20000,  "mult": 0.6},
        {"max_chars": 50000,  "mult": 0.85},
        {"max_chars": 80000,  "mult": 1.0},
        {"max_chars": 150000, "mult": 1.4},
        {"max_chars": None,   "mult": 1.8},
    ],
    "model_multipliers": {
        "grok": 0.8, "gpt-5.4": 1.0, "gpt-5.5": 1.3,
        "mistral-small": 0.8, "mistral-large": 1.0, "mistral-medium-3-5": 1.7,
        "gemini": 4.0, "sonar": 3.0, "default": 1.0,
    },
    "effort_multipliers": {
        "none": 1.0, "low": 1.15, "medium": 2.0, "high": 3.5, "xhigh": 10.5,
        "default": 1.0,
    },
}


def _load():
    path = Path(__file__).parent / "configs" / "timeouts.yaml"
    if not path.exists():
        return _FALLBACK
    try:
        with open(path, encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
        # Minimal shape validation — fall back if any required section is missing.
        for key in ("base_seconds", "floor_seconds", "size_multipliers",
                    "model_multipliers", "effort_multipliers"):
            if key not in data:
                raise ValueError(f"timeouts.yaml missing '{key}'")
        return data
    except Exception as exc:
        log.warning("Could not load configs/timeouts.yaml (%s) — using built-in defaults", exc)
        return _FALLBACK


_CONFIG = _load()


def _size_mult(char_count, buckets):
    for b in buckets:
        cap = b.get("max_chars")
        if cap is None or char_count <= cap:
            return float(b["mult"])
    return float(buckets[-1]["mult"])


def _model_mult(model_id, table):
    if not model_id:
        return float(table.get("default", 1.0))
    # Longest key first so "mistral-medium-3-5" wins over a hypothetical "mistral".
    for key in sorted((k for k in table if k != "default"), key=len, reverse=True):
        if model_id.startswith(key):
            return float(table[key])
    return float(table.get("default", 1.0))


def compute_timeout(char_count, model_id, effort, task_ceiling_seconds, config=None):
    """Return the effective per-call timeout in seconds for one model.

    char_count:           length of the draft text.
    model_id:             resolved model id (e.g. "gpt-5.5").
    effort:               reasoning_effort / effort value, or None.
    task_ceiling_seconds: absolute upper bound; result is clamped to ceiling − 15.
    """
    cfg = config or _CONFIG
    base = float(cfg["base_seconds"])
    floor = int(cfg["floor_seconds"])
    size = _size_mult(char_count, cfg["size_multipliers"])
    model = _model_mult(model_id, cfg["model_multipliers"])
    eff_table = cfg["effort_multipliers"]
    eff = float(eff_table.get(effort or "default", eff_table.get("default", 1.0)))
    variance = float(cfg.get("variance_margin", 1.0))  # default 1.0 keeps old behavior if absent

    raw = base * size * model * eff * variance
    ceiling = max(floor, int(task_ceiling_seconds) - 15)
    return int(min(max(raw, floor), ceiling))


def compute_all(char_count, model_configs, task_ceiling_seconds, config=None):
    """Compute effective timeouts for every enabled model.

    An explicit ``timeout_seconds`` already present on a model config is treated as
    a user override and returned unchanged. All other models get a computed value.
    Returns ``{provider: timeout_seconds}``.
    """
    out = {}
    for provider, cfg in (model_configs or {}).items():
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            continue
        explicit = cfg.get("timeout_seconds")
        if explicit is not None:
            out[provider] = int(explicit)
            continue
        effort = cfg.get("reasoning_effort") or cfg.get("effort")
        out[provider] = compute_timeout(
            char_count, cfg.get("model", ""), effort, task_ceiling_seconds, config=config
        )
    return out
