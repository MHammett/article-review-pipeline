"""Merge per-model synthesis outputs into a consolidated style profile.

consolidate_lists  — weighted threshold filtering for banned_words/phrases/rules
collect_prose      — gather prose outputs sorted by model weight
consolidate_detection — reconcile multi-model detection outputs via Claude API call
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from ci_core.llm.json_utils import extract_json
from .callers import call_one

log = logging.getLogger(__name__)

# Style weights copied from consolidation.py (_DEFAULT_WEIGHTS, voice_style domain)
DEFAULT_STYLE_WEIGHTS: dict[str, float] = {
    "openai": 1.2,
    "claude": 1.1,
    "mistral": 1.0,
    "gemini": 1.0,
    "grok": 1.0,
    "perplexity": 1.0,
}

DEFAULT_CONSENSUS_THRESHOLD = 2.0


def _get_weight(model_name: str) -> float:
    return DEFAULT_STYLE_WEIGHTS.get(model_name, 1.0)


def consolidate_lists(
    results: dict[str, dict],
    key: str,
    threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
    weights: dict[str, float] | None = None,
) -> list[str]:
    """Return items from `results[model][key]` whose weighted vote sum meets threshold.

    Items present only in low-weight models are dropped. Items agreed upon by
    Claude + OpenAI (combined weight 2.3) always pass the 2.0 threshold.
    """
    w = weights or DEFAULT_STYLE_WEIGHTS
    item_weights: dict[str, float] = defaultdict(float)

    for model_name, result in results.items():
        if result.get("failed"):
            continue
        parsed = result.get("_parsed") or {}
        items = parsed.get(key, []) or []
        model_weight = w.get(model_name, 1.0)
        for item in items:
            if isinstance(item, str) and item.strip():
                item_weights[item.strip()] += model_weight

    return [
        item
        for item, score in sorted(item_weights.items(), key=lambda x: -x[1])
        if score >= threshold
    ]


def collect_prose(
    results: dict[str, dict],
    key: str,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Collect prose outputs sorted by model weight descending.

    Returns list of {"model": str, "weight": float, "text": str}.
    Used for style_profile and audience_* prose passed to reconciliation.
    """
    w = weights or DEFAULT_STYLE_WEIGHTS
    entries = []
    for model_name, result in results.items():
        if result.get("failed"):
            continue
        parsed = result.get("_parsed") or {}
        text = parsed.get(key)
        if text and isinstance(text, str):
            entries.append(
                {"model": model_name, "weight": w.get(model_name, 1.0), "text": text}
            )
    return sorted(entries, key=lambda x: -x["weight"])


def consolidate_detection(
    detection_results: dict[str, dict],
    user_config: dict,
    weights: dict[str, float] | None = None,
) -> tuple[list[Any], Any] | None:
    """Reconcile per-model detection outputs into a unified cluster list.

    Makes ONE Claude API call using prompts/consolidate_detection.txt.
    On success returns ``(raw_styles, overall_confidence)`` where ``raw_styles``
    is a list of StyleCluster-shaped dicts (caller converts to StyleCluster
    objects). Returns ``None`` if consolidation could not be performed.
    """
    from ci_core.config_helpers import normalize_model_configs

    w = weights or DEFAULT_STYLE_WEIGHTS

    # Build input for the consolidation prompt
    model_sections = []
    for model_name, result in detection_results.items():
        if result.get("failed"):
            continue
        content = result.get("content", "")
        model_weight = w.get(model_name, 1.0)
        model_sections.append(
            f"--- {model_name} (weight: {model_weight}) ---\n{content}"
        )

    if not model_sections:
        log.warning(
            "consolidate_detection: all detection models failed; returning empty cluster list"
        )
        return None

    user_prompt = "\n\n".join(model_sections)

    prompt_path = Path(__file__).parent / "prompts" / "consolidate_detection.txt"
    try:
        system_prompt = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.error("consolidate_detection.txt not found at %s", prompt_path)
        return None

    # Always use Claude for reconciliation
    models_cfg = normalize_model_configs(user_config.get("models", {}))
    claude_cfg = models_cfg.get("claude")
    if not claude_cfg:
        log.error(
            "consolidate_detection: Claude not configured; cannot reconcile detection outputs"
        )
        return None

    api_keys = user_config.get("api_keys", {})
    result = call_one(
        model_name="claude",
        model_cfg=claude_cfg,
        api_keys=api_keys,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pass_name="consolidate_detection",
    )

    if result.get("failed"):
        log.error("consolidate_detection: Claude call failed: %s", result.get("error"))
        return None

    parsed = extract_json(result.get("content", ""))
    if not parsed:
        log.error("consolidate_detection: could not parse Claude response as JSON")
        return None

    styles = parsed.get("detected_styles", [])
    if not isinstance(styles, list):
        log.error("consolidate_detection: response missing 'detected_styles' list")
        return None

    overall_confidence = parsed.get("overall_confidence", "medium")
    notes = parsed.get("consolidation_notes", "")
    if notes:
        log.info("Detection consolidation notes: %s", notes)

    return styles, overall_confidence
