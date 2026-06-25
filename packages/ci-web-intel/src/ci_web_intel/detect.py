"""Voice cluster detection and document classification.

detect_voices       — detection pass: calls models, consolidates, returns VoiceCluster list
classify_documents  — metric-based classification, no API calls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.review.json_utils import extract_json
from callers import call_all, call_one
from collectors.base import Document
from voice_consolidation import DEFAULT_VOICE_WEIGHTS, consolidate_detection

log = logging.getLogger(__name__)


class CanonicalFallbackWarning(Warning):
    """Raised when detection confidence is too low to use detect mode."""


@dataclass
class VoiceCluster:
    label: str
    description: str
    features: dict
    source_distribution: dict
    sample_ids: list[str]
    assigned_docs: list[Document] = field(default_factory=list)
    word_count: int = 0
    confidence: str = "medium"


def _build_detection_sample(docs: list[Document], max_chars: int) -> str:
    """Build a stratified sample across sources for the detection pass."""
    # Group by source
    by_source: dict[str, list[Document]] = {}
    for doc in docs:
        by_source.setdefault(doc.source, []).append(doc)

    # Sort each source by date (most recent first)
    for src in by_source:
        by_source[src].sort(key=lambda d: d.date or "", reverse=True)

    # Interleave sources proportionally
    sources = list(by_source.keys())
    indices = {src: 0 for src in sources}
    result_parts = []
    total_chars = 0

    while True:
        added = False
        for src in sources:
            idx = indices[src]
            docs_for_src = by_source[src]
            if idx >= len(docs_for_src):
                continue
            doc = docs_for_src[idx]
            indices[src] += 1

            # Add a labeled excerpt
            snippet = f"[SOURCE: {doc.source} | ID: {doc.url_or_id} | DATE: {doc.date}]\n{doc.text[:3000]}\n---\n"
            if total_chars + len(snippet) > max_chars:
                break
            result_parts.append(snippet)
            total_chars += len(snippet)
            added = True

        if not added:
            break

    return "\n".join(result_parts)


def detect_voices(
    docs: list[Document],
    user_config: dict,
    max_voices: int = 5,
    detection_models: list[str] | str | None = None,
    max_parallel: int = 0,
    max_chars: int = 80000,
) -> list[VoiceCluster]:
    """Run detection pass: stratified sample → detection models → VoiceCluster list.

    detection_models:
        None / []  = voice-style-weighted subset (claude + openai by default)
        "*"        = all configured models
        list[str]  = explicit list

    Raises CanonicalFallbackWarning if overall_confidence is "low".
    Returns empty list if all detection models fail (caller falls back to canonical).
    """
    # Determine which models to use for detection
    if detection_models == "*":
        model_subset = None  # all configured
    elif not detection_models:
        # Voice-style-weighted subset: claude + openai (weight ≥ 1.1)
        from config_loader import _normalize_model_configs
        configured = set(_normalize_model_configs(user_config.get("models", {})).keys())
        model_subset = [m for m in ("claude", "openai") if m in configured]
        if not model_subset:
            model_subset = None
    else:
        model_subset = list(detection_models)

    sample = _build_detection_sample(docs, max_chars)
    if not sample:
        log.warning("detect_voices: corpus is empty after sampling")
        return []

    prompt_path = Path(__file__).parent / "prompts" / "detect_voices.txt"
    try:
        system_prompt = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.error("detect_voices.txt not found at %s", prompt_path)
        return []

    max_voices_str = str(max_voices) if max_voices > 0 else "no limit"
    user_prompt = (
        f"Maximum voices to identify: {max_voices_str}\n\n"
        f"Writing corpus (stratified sample):\n\n{sample}"
    )

    log.info("Detection pass: calling %s", model_subset or "voice-style-weighted models")
    detection_results = call_all(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        user_config=user_config,
        models=model_subset,
        max_parallel=max_parallel,
        exclude_perplexity=True,
        pass_name="detect_voices",
    )

    # Parse each model's detection output
    for model_name, result in detection_results.items():
        if not result.get("failed"):
            parsed = extract_json(result.get("content", ""))
            result["_parsed"] = parsed or {}

    # Check if ALL models failed
    any_succeeded = any(not r.get("failed") for r in detection_results.values())
    if not any_succeeded:
        log.error("detect_voices: all detection models failed; falling back to canonical mode")
        return []

    # If only one model, use its output directly; otherwise consolidate
    successful = {k: v for k, v in detection_results.items() if not v.get("failed")}

    if len(successful) == 1:
        model_name, result = next(iter(successful.items()))
        parsed = result.get("_parsed") or {}
        raw_voices = parsed.get("detected_voices", [])
        overall_confidence = parsed.get("overall_confidence", "medium")
        notes = parsed.get("detection_notes", "")
        if notes:
            log.info("Detection notes (%s): %s", model_name, notes)
    else:
        result_pair = consolidate_detection(detection_results, user_config)
        if not result_pair:
            return []
        raw_voices, overall_confidence = result_pair

    if overall_confidence == "low":
        log.warning("Detection confidence is 'low' — falling back to canonical synthesis mode")
        raise CanonicalFallbackWarning(
            "Detection pass returned overall_confidence='low'. "
            "Corpus may be too small or too homogeneous. Falling back to canonical mode."
        )

    # Apply max_voices ceiling
    if max_voices > 0:
        raw_voices = raw_voices[:max_voices]

    clusters = []
    for v in raw_voices:
        if not isinstance(v, dict) or not v.get("label"):
            continue
        cluster = VoiceCluster(
            label=v.get("label", "unknown"),
            description=v.get("description", ""),
            features=v.get("features", {}),
            source_distribution=v.get("source_distribution", {}),
            sample_ids=v.get("sample_ids", []),
            confidence=v.get("confidence", "medium"),
        )
        clusters.append(cluster)

    log.info("Detection: found %d voice cluster(s): %s", len(clusters), [c.label for c in clusters])
    return clusters


def _score_doc_for_cluster(doc: Document, features: dict) -> float:
    """Score a document against a cluster's metric thresholds. Higher = better match."""
    metrics = doc.metrics
    if not metrics or not features:
        return 0.0

    score = 0.0
    matched = 0
    for metric_name, rule in features.items():
        if metric_name not in metrics:
            continue
        try:
            op, threshold = rule[0], rule[1]
            value = metrics[metric_name]
            if op == ">" and value > threshold:
                score += 1.0
            elif op == "<" and value < threshold:
                score += 1.0
            matched += 1
        except (IndexError, TypeError, KeyError):
            continue

    return score / max(matched, 1)


def _feature_distance(features_a: dict, features_b: dict) -> float:
    """Simple distance between two feature dicts (for cluster merging)."""
    all_keys = set(features_a) | set(features_b)
    if not all_keys:
        return 0.0
    diff = 0.0
    for key in all_keys:
        a_rule = features_a.get(key, [">", 0])
        b_rule = features_b.get(key, [">", 0])
        try:
            diff += abs(float(a_rule[1]) - float(b_rule[1]))
        except (IndexError, TypeError, ValueError):
            diff += 1.0
    return diff / len(all_keys)


def classify_documents(
    docs: list[Document],
    clusters: list[VoiceCluster],
    ambiguity_threshold: float = 0.2,
    per_voice_min_words: int = 2000,
) -> tuple[dict[str, list[Document]], list[Document]]:
    """Assign documents to clusters using pre-computed metrics. No API calls.

    Returns:
        (cluster_label → assigned docs, ambiguous_docs)
    Clusters below per_voice_min_words are merged into the nearest cluster.
    """
    if not clusters:
        return {}, list(docs)

    cluster_docs: dict[str, list[Document]] = {c.label: [] for c in clusters}
    ambiguous_docs: list[Document] = []

    for doc in docs:
        scores = [(c.label, _score_doc_for_cluster(doc, c.features)) for c in clusters]
        scores.sort(key=lambda x: -x[1])

        if len(scores) < 2:
            top_label = scores[0][0] if scores else clusters[0].label
            cluster_docs[top_label].append(doc)
        else:
            top_label, top_score = scores[0]
            second_label, second_score = scores[1]
            if top_score - second_score < ambiguity_threshold:
                ambiguous_docs.append(doc)
            else:
                cluster_docs[top_label].append(doc)

    # Populate assigned_docs and word_count on each cluster
    for cluster in clusters:
        assigned = cluster_docs.get(cluster.label, [])
        cluster.assigned_docs = assigned
        cluster.word_count = sum(d.word_count for d in assigned)

    # Merge undersized clusters
    changed = True
    while changed:
        changed = False
        small = [c for c in clusters if c.word_count < per_voice_min_words]
        if not small:
            break
        for small_cluster in small:
            if len(clusters) <= 1:
                break
            # Find nearest cluster by feature distance
            others = [c for c in clusters if c.label != small_cluster.label]
            if not others:
                break
            nearest = min(others, key=lambda c: _feature_distance(small_cluster.features, c.features))
            log.warning(
                "Cluster %r has only %d words (< %d minimum); merging into %r",
                small_cluster.label, small_cluster.word_count, per_voice_min_words, nearest.label,
            )
            # Merge docs
            nearest.assigned_docs.extend(small_cluster.assigned_docs)
            nearest.word_count += small_cluster.word_count
            cluster_docs[nearest.label].extend(cluster_docs.pop(small_cluster.label, []))
            clusters.remove(small_cluster)
            changed = True
            break

    # Log stats
    log.info("Document classification results:")
    for cluster in clusters:
        src_summary = ", ".join(
            f"primarily {s}"
            for s, _ in sorted(cluster.source_distribution.items(), key=lambda x: -x[1])[:2]
        ) or ""
        log.info(
            "  %-25s  %4d docs  (%6d words)  %s",
            cluster.label + ":", len(cluster.assigned_docs), cluster.word_count, src_summary,
        )
    log.info("  %-25s  %4d docs", "ambiguous / canonical:", len(ambiguous_docs))

    return cluster_docs, ambiguous_docs
