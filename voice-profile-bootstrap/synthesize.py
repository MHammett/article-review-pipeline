"""Orchestrates multi-model synthesis for all three voice modes.

Modes:
  canonical   — M+1 API calls (M synthesis + 1 Claude reconciliation)
  detect      — D+1+(N×M)+1 (detect + consolidate + per-voice×models + reconcile)
  per-source  — (G×M)+1 (source groups × models + 1 reconciliation)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from adapters.review.json_utils import extract_json
from callers import call_all, call_one
from collectors.base import Document
from detect import VoiceCluster, CanonicalFallbackWarning, classify_documents, detect_voices
from voice_consolidation import collect_prose, consolidate_lists, DEFAULT_VOICE_WEIGHTS

log = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "prompts"


class SynthesisError(Exception):
    pass


def _load_prompt(name: str) -> str:
    path = _PROMPT_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SynthesisError(f"Prompt file not found: {path}")


def _sample_docs(docs: list[Document], max_chars: int, label: str = "") -> str:
    """Sample documents (most-recent first) within max_chars budget."""
    sorted_docs = sorted(docs, key=lambda d: d.date or "", reverse=True)
    parts = []
    total = 0
    for doc in sorted_docs:
        snippet = f"[ID: {doc.url_or_id} | DATE: {doc.date} | SOURCE: {doc.source}]\n{doc.text}\n---\n"
        if total + len(snippet) > max_chars:
            log.debug("Corpus budget reached for %s; sampled %d/%d docs", label or "corpus", len(parts), len(docs))
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n".join(parts)


def _parse_synthesis_result(result: dict, model_name: str, pass_label: str) -> dict | None:
    """Parse and attach _parsed to result dict; return parsed dict or None."""
    content = result.get("content", "")
    parsed = extract_json(content)
    if parsed is None:
        log.warning("%s: could not extract JSON from %s response", pass_label, model_name)
        return None
    result["_parsed"] = parsed
    return parsed


def _build_reconcile_input(
    results: dict[str, dict],
    consolidated_lists: dict,
    voice_label: str = "",
) -> str:
    """Build user prompt for the reconciliation pass."""
    sections = []
    for model_name, result in results.items():
        if result.get("failed"):
            continue
        content = result.get("content", "")
        weight = DEFAULT_VOICE_WEIGHTS.get(model_name, 1.0)
        sections.append(f"--- {model_name} (weight: {weight}) ---\n{content}")

    combined_lists = (
        f"--- consolidated_lists ---\n"
        f"banned_words: {consolidated_lists.get('banned_words', [])}\n"
        f"banned_phrases: {consolidated_lists.get('banned_phrases', [])}\n"
        f"positive_rules: {consolidated_lists.get('positive_rules', [])}\n"
    )

    header = f"Voice cluster: {voice_label}\n\n" if voice_label else ""
    return header + "\n\n".join(sections) + "\n\n" + combined_lists


def _run_synthesis_pass(
    docs: list[Document],
    user_config: dict,
    synthesis_models: list[str] | None,
    system_prompt_file: str,
    corpus_budget: int,
    max_parallel: int,
    label: str,
    extra_context: str = "",
) -> dict[str, dict]:
    """Run synthesis for a set of docs against a prompt; parse all results."""
    system_prompt = _load_prompt(system_prompt_file)
    corpus_text = _sample_docs(docs, corpus_budget, label)
    user_prompt = (extra_context + "\n\n" if extra_context else "") + corpus_text

    results = call_all(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        user_config=user_config,
        models=synthesis_models,
        max_parallel=max_parallel,
        pass_name=f"synthesize_{label}",
    )

    for model_name, result in results.items():
        if not result.get("failed"):
            _parse_synthesis_result(result, model_name, label)

    return results


def _reconcile(
    results: dict[str, dict],
    consolidated_lists: dict,
    user_config: dict,
    pass_name: str,
    voice_label: str = "",
) -> dict:
    """Run Claude reconciliation pass. Returns parsed result dict."""
    from config_loader import _normalize_model_configs
    system_prompt = _load_prompt("synthesize_reconcile.txt")
    user_prompt = _build_reconcile_input(results, consolidated_lists, voice_label)

    # Always use Claude for reconciliation
    models_cfg = _normalize_model_configs(user_config.get("models", {}))
    claude_cfg = models_cfg.get("claude")
    if not claude_cfg:
        raise SynthesisError("Claude not configured; required for reconciliation pass")

    api_keys = user_config.get("api_keys", {})
    result = call_one(
        model_name="claude",
        model_cfg=claude_cfg,
        api_keys=api_keys,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pass_name=pass_name,
    )
    if result.get("failed"):
        raise SynthesisError(f"Reconciliation failed: {result.get('error')}")

    parsed = extract_json(result.get("content", ""))
    if not parsed:
        raise SynthesisError("Reconciliation returned unparseable response")
    return parsed


def validate_synthesis_output(raw: dict, mode: str) -> dict:
    """Validate required keys; raise SynthesisError on missing."""
    if mode == "canonical":
        required = ["voice_profile", "audience_primary", "banned_words", "banned_phrases", "positive_rules"]
        missing = [k for k in required if k not in raw or raw[k] is None]
        if missing:
            raise SynthesisError(f"Canonical profile missing required keys: {missing}")
        return raw

    # detect / per-source
    if "canonical" not in raw or raw["canonical"] is None:
        raise SynthesisError("Missing top-level keys: ['canonical']")

    canonical = raw["canonical"]
    canonical_required = ["voice_profile", "banned_words", "banned_phrases", "positive_rules"]
    missing_canon = [k for k in canonical_required if k not in canonical or canonical[k] is None]
    if missing_canon:
        raise SynthesisError(f"Canonical section missing: {missing_canon}")
    return raw


def _canonical_mode(
    docs: list[Document],
    user_config: dict,
    synthesis_models: list[str] | None,
    corpus_budget: int,
    max_parallel: int,
) -> dict:
    log.info("Synthesis: canonical mode — %d docs, %d chars budget", len(docs), corpus_budget)
    results = _run_synthesis_pass(
        docs, user_config, synthesis_models,
        "synthesize_canonical.txt", corpus_budget, max_parallel, "canonical",
    )

    any_ok = any(not r.get("failed") for r in results.values())
    if not any_ok:
        raise SynthesisError("no model produced a valid synthesis result")

    consolidated = {
        "banned_words": consolidate_lists(results, "banned_words"),
        "banned_phrases": consolidate_lists(results, "banned_phrases"),
        "positive_rules": consolidate_lists(results, "positive_rules"),
    }

    reconciled = _reconcile(results, consolidated, user_config, "reconcile_canonical")
    canonical = reconciled.get("canonical") or reconciled
    # Inject consolidated lists (reconcile prompt says to use them verbatim)
    canonical["banned_words"] = consolidated["banned_words"]
    canonical["banned_phrases"] = consolidated["banned_phrases"]
    canonical["positive_rules"] = consolidated["positive_rules"]

    validate_synthesis_output(canonical, "canonical")
    return canonical


def _detect_mode(
    docs: list[Document],
    ambiguous_docs: list[Document],
    clusters: list[VoiceCluster],
    user_config: dict,
    synthesis_models: list[str] | None,
    corpus_budget: int,
    max_parallel: int,
) -> dict:
    """Synthesize per-voice profiles and reconcile into canonical + detected_voices."""
    log.info("Synthesis: detect mode — %d clusters, %d docs", len(clusters), len(docs))

    all_docs_for_canonical = docs + ambiguous_docs

    # Canonical synthesis (full corpus including ambiguous)
    canonical_results = _run_synthesis_pass(
        all_docs_for_canonical, user_config, synthesis_models,
        "synthesize_canonical.txt", corpus_budget, max_parallel, "canonical",
    )

    canonical_consolidated = {
        "banned_words": consolidate_lists(canonical_results, "banned_words"),
        "banned_phrases": consolidate_lists(canonical_results, "banned_phrases"),
        "positive_rules": consolidate_lists(canonical_results, "positive_rules"),
    }

    # Per-voice synthesis
    per_voice_results: dict[str, dict] = {}
    for cluster in clusters:
        cluster_docs = cluster.assigned_docs
        if not cluster_docs:
            log.warning("Cluster %r has no assigned docs; skipping", cluster.label)
            continue

        extra = (
            f"Voice cluster: {cluster.label}\n"
            f"Description: {cluster.description}\n"
            f"Source distribution: {cluster.source_distribution}\n"
        )
        voice_results = _run_synthesis_pass(
            cluster_docs, user_config, synthesis_models,
            "synthesize_per_voice.txt",
            corpus_budget // max(len(clusters), 1),
            max_parallel,
            f"voice_{cluster.label}",
            extra_context=extra,
        )

        any_ok = any(not r.get("failed") for r in voice_results.values())
        if not any_ok:
            log.warning("All models failed for voice %r; skipping", cluster.label)
            continue

        per_voice_results[cluster.label] = {
            "results": voice_results,
            "consolidated": {
                "banned_words": consolidate_lists(voice_results, "additional_banned_words"),
                "positive_rules": consolidate_lists(voice_results, "additional_positive_rules"),
            },
            "cluster": cluster,
        }

    # Build reconciliation input
    reconcile_results = dict(canonical_results)
    for voice_label, voice_data in per_voice_results.items():
        for model_name, r in voice_data["results"].items():
            reconcile_results[f"{model_name}:{voice_label}"] = r

    reconciled = _reconcile(
        canonical_results, canonical_consolidated, user_config, "reconcile_detect",
    )

    # Ensure structure
    canonical_out = reconciled.get("canonical") or reconciled
    canonical_out["banned_words"] = canonical_consolidated["banned_words"]
    canonical_out["banned_phrases"] = canonical_consolidated["banned_phrases"]
    canonical_out["positive_rules"] = canonical_consolidated["positive_rules"]

    # Build detected_voices from per-voice data
    detected_voices_out: dict[str, dict] = {}
    recon_detected = reconciled.get("detected_voices") or {}

    for voice_label, voice_data in per_voice_results.items():
        cluster = voice_data["cluster"]
        voice_results = voice_data["results"]
        additional = voice_data["consolidated"]

        # Pick best prose from reconciliation or model outputs
        recon_voice = recon_detected.get(voice_label) or {}
        prose_entries = collect_prose(voice_results, "voice_profile")
        voice_profile_text = recon_voice.get("voice_profile") or (prose_entries[0]["text"] if prose_entries else "")

        detected_voices_out[voice_label] = {
            "voice_profile": voice_profile_text,
            "additional_banned_words": recon_voice.get("additional_banned_words") or additional["banned_words"],
            "additional_positive_rules": recon_voice.get("additional_positive_rules") or additional["positive_rules"],
            "voice_notes": recon_voice.get("voice_notes") or "",
            "source_distribution": cluster.source_distribution,
            "doc_count": len(cluster.assigned_docs),
            "confidence": cluster.confidence,
        }

    notes = reconciled.get("synthesis_notes", "")
    if notes:
        log.info("Synthesis notes: %s", notes)

    return {
        "canonical": canonical_out,
        "detected_voices": detected_voices_out,
        "synthesis_notes": notes,
    }


def _per_source_mode(
    docs: list[Document],
    user_config: dict,
    synthesis_models: list[str] | None,
    corpus_budget: int,
    max_parallel: int,
    group_by: str = "source",
) -> dict:
    """Synthesize per source group, then reconcile."""
    # Group documents
    groups: dict[str, list[Document]] = {}
    for doc in docs:
        key = doc.source if group_by == "source" else doc.register
        groups.setdefault(key, []).append(doc)

    log.info("Synthesis: per-source mode (%d groups: %s)", len(groups), list(groups.keys()))

    canonical_results = _run_synthesis_pass(
        docs, user_config, synthesis_models,
        "synthesize_canonical.txt", corpus_budget, max_parallel, "canonical",
    )

    canonical_consolidated = {
        "banned_words": consolidate_lists(canonical_results, "banned_words"),
        "banned_phrases": consolidate_lists(canonical_results, "banned_phrases"),
        "positive_rules": consolidate_lists(canonical_results, "positive_rules"),
    }

    per_source_results: dict[str, dict] = {}
    for group_label, group_docs in groups.items():
        extra = f"Source group: {group_label}\nDocuments from: {group_label}\n"
        group_results = _run_synthesis_pass(
            group_docs, user_config, synthesis_models,
            "synthesize_per_source.txt",
            corpus_budget // max(len(groups), 1),
            max_parallel,
            f"source_{group_label}",
            extra_context=extra,
        )

        any_ok = any(not r.get("failed") for r in group_results.values())
        if not any_ok:
            log.warning("All models failed for source group %r; skipping", group_label)
            continue

        per_source_results[group_label] = {
            "results": group_results,
            "consolidated": {
                "banned_words": consolidate_lists(group_results, "additional_banned_words"),
                "positive_rules": consolidate_lists(group_results, "additional_positive_rules"),
            },
            "docs": group_docs,
        }

    reconciled = _reconcile(
        canonical_results, canonical_consolidated, user_config, "reconcile_per_source",
    )

    canonical_out = reconciled.get("canonical") or reconciled
    canonical_out["banned_words"] = canonical_consolidated["banned_words"]
    canonical_out["banned_phrases"] = canonical_consolidated["banned_phrases"]
    canonical_out["positive_rules"] = canonical_consolidated["positive_rules"]

    detected_voices_out: dict[str, dict] = {}
    recon_detected = reconciled.get("detected_voices") or {}

    for group_label, group_data in per_source_results.items():
        group_docs_list = group_data["docs"]
        group_results = group_data["results"]
        additional = group_data["consolidated"]
        recon_voice = recon_detected.get(group_label) or {}
        prose_entries = collect_prose(group_results, "voice_profile")
        voice_profile_text = recon_voice.get("voice_profile") or (prose_entries[0]["text"] if prose_entries else "")

        detected_voices_out[group_label] = {
            "voice_profile": voice_profile_text,
            "additional_banned_words": additional["banned_words"],
            "additional_positive_rules": additional["positive_rules"],
            "voice_notes": recon_voice.get("voice_notes") or "",
            "source_distribution": {group_label: 1.0},
            "doc_count": len(group_docs_list),
            "confidence": "medium",
        }

    notes = reconciled.get("synthesis_notes", "")
    if notes:
        log.info("Synthesis notes: %s", notes)

    return {
        "canonical": canonical_out,
        "detected_voices": detected_voices_out,
        "synthesis_notes": notes,
    }


def synthesize(
    docs: list[Document],
    user_config: dict,
    mode: str = "detect",
    synthesis_models: list[str] | None = None,
    detection_models: list[str] | str | None = None,
    max_voices: int = 5,
    max_input_chars: int = 120000,
    prompt_overhead_chars: int = 12000,
    ambiguity_threshold: float = 0.2,
    per_voice_min_words: int = 2000,
    max_parallel: int = 0,
    per_source_group_by: str = "source",
    synthesis_config: dict | None = None,
) -> dict:
    """Main entry point for synthesis. Orchestrates all three modes.

    Returns a validated profile dict.
    """
    corpus_budget = max_input_chars - prompt_overhead_chars
    cfg = synthesis_config or {}

    # Pre-synthesis call count estimate
    n_models = len(synthesis_models) if synthesis_models else len(
        [k for k, v in user_config.get("models", {}).items()
         if isinstance(v, dict) and v.get("enabled", True) is not False or isinstance(v, str)]
    )

    if mode == "canonical":
        log.info(
            "Pre-synthesis: canonical mode — ~%d synthesis + 1 reconciliation = ~%d calls",
            n_models, n_models + 1,
        )
        result = _canonical_mode(docs, user_config, synthesis_models, corpus_budget, max_parallel)
        validate_synthesis_output(result, "canonical")
        return result

    elif mode == "detect":
        n_detection = 2  # estimated (claude + openai by default)
        log.info(
            "Pre-synthesis: detect mode — ~%d detection + 1 consolidation + N×%d synthesis + 1 reconciliation",
            n_detection, n_models,
        )
        try:
            clusters = detect_voices(
                docs, user_config,
                max_voices=max_voices,
                detection_models=detection_models,
                max_parallel=max_parallel,
                max_chars=corpus_budget,
            )
        except CanonicalFallbackWarning as w:
            log.warning("Falling back to canonical mode: %s", w)
            result = _canonical_mode(docs, user_config, synthesis_models, corpus_budget, max_parallel)
            result["_fallback_reason"] = str(w)
            validate_synthesis_output(result, "canonical")
            return result

        if not clusters:
            log.warning("No clusters detected; falling back to canonical mode")
            result = _canonical_mode(docs, user_config, synthesis_models, corpus_budget, max_parallel)
            result["_fallback_reason"] = "No clusters detected"
            validate_synthesis_output(result, "canonical")
            return result

        cluster_map, ambiguous_docs = classify_documents(
            docs, clusters, ambiguity_threshold, per_voice_min_words,
        )

        result = _detect_mode(
            docs, ambiguous_docs, clusters,
            user_config, synthesis_models, corpus_budget, max_parallel,
        )
        validate_synthesis_output(result, "detect")
        return result

    elif mode == "per-source":
        n_groups = len(set(
            doc.source if per_source_group_by == "source" else doc.register
            for doc in docs
        ))
        log.info(
            "Pre-synthesis: per-source mode — ~%d groups × %d models + 1 reconciliation = ~%d calls",
            n_groups, n_models, n_groups * n_models + 1,
        )
        result = _per_source_mode(
            docs, user_config, synthesis_models, corpus_budget, max_parallel, per_source_group_by,
        )
        validate_synthesis_output(result, "detect")
        return result

    else:
        raise SynthesisError(f"Unknown synthesis mode: {mode!r}")
