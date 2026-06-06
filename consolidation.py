"""
Merge five model responses into a structured review report.
Identifies consensus flags (3+ models, or 2+ models + LanguageTool hit).
"""
from datetime import datetime, timezone


def _passage_key(passage):
    """Normalize a passage for fuzzy cross-model matching."""
    return " ".join(passage.lower().split())[:120]


def _find_consensus(model_flags, lt_flagged_passages, threshold=3):
    """
    model_flags: dict of {model_name: [{"passage": ..., ...}, ...]}
    Returns (consensus_flags, single_model_flags)
    """
    passage_to_models = {}

    for model, flags in model_flags.items():
        for flag in flags:
            key = _passage_key(flag.get("passage", ""))
            if not key:
                continue
            if key not in passage_to_models:
                passage_to_models[key] = {"models": [], "flag_data": []}
            passage_to_models[key]["models"].append(model)
            passage_to_models[key]["flag_data"].append({**flag, "source_model": model})

    # Check LanguageTool overlap
    lt_keys = {_passage_key(p) for p in lt_flagged_passages}

    consensus = []
    single_model = []

    for key, entry in passage_to_models.items():
        models = entry["models"]
        has_lt = key in lt_keys
        count = len(set(models))  # unique models
        lt_boost = 1 if has_lt else 0

        if count + lt_boost >= threshold or (count >= 2 and has_lt):
            consensus.append({
                "passage": entry["flag_data"][0].get("passage", ""),
                "models": list(set(models)),
                "languagetool_also_flagged": has_lt,
                "flags": entry["flag_data"],
            })
        else:
            for fd in entry["flag_data"]:
                single_model.append(fd)

    return consensus, single_model


def build_report(
    article_title,
    publication_name,
    run_number,
    corrected_draft,
    lt_result,
    gemini_result,
    openai_voice_result,
    mistral_argument_result,
    openai_completeness_result,
    mistral_redteam_result,
    api_call_log,
    prior_report=None,
):
    now = datetime.now(timezone.utc).isoformat()

    # Collect flags by model
    model_flags = {}

    def _extract_flags(result, model_key):
        if result.get("failed") or not result.get("data"):
            return []
        return result["data"].get("flags", [])

    openai_voice_flags = _extract_flags(openai_voice_result, "openai_voice")
    mistral_arg_flags = _extract_flags(mistral_argument_result, "mistral_argument")
    openai_comp_flags = _extract_flags(openai_completeness_result, "openai_completeness")

    if openai_voice_flags:
        model_flags["openai_voice"] = openai_voice_flags
    if mistral_arg_flags:
        model_flags["mistral_argument"] = mistral_arg_flags
    if openai_comp_flags:
        # completeness flags use "what_is_missing" not "passage"; normalize
        normalized = []
        for f in openai_comp_flags:
            normalized.append({
                "passage": f.get("passage_reference", ""),
                **f,
            })
        model_flags["openai_completeness"] = normalized

    # LanguageTool flagged passages (flag_for_review, not auto-applied)
    lt_flagged_passages = []
    if lt_result and not lt_result.get("failed"):
        for match in lt_result.get("flagged_matches", []):
            ctx = match.get("context", "")
            if ctx:
                lt_flagged_passages.append(ctx)

    consensus_flags, single_model_flags = _find_consensus(model_flags, lt_flagged_passages)

    # Low-confidence flags from each model
    low_confidence = []
    for result_name, result in [
        ("openai_voice", openai_voice_result),
        ("mistral_argument", mistral_argument_result),
        ("openai_completeness", openai_completeness_result),
    ]:
        if not result.get("failed") and result.get("data"):
            for lc in result["data"].get("low_confidence", []):
                low_confidence.append({**lc, "source_model": result_name})

    # Fact check section
    fact_check = {}
    if not gemini_result.get("failed") and gemini_result.get("data"):
        fact_check = gemini_result["data"]

    # Red team section
    red_team = {}
    if not mistral_redteam_result.get("failed") and mistral_redteam_result.get("data"):
        red_team = mistral_redteam_result["data"]

    # Delta from prior run
    delta = None
    if prior_report:
        delta = _compute_delta(corrected_draft, prior_report, consensus_flags)

    report = {
        "generated": now,
        "run_number": run_number,
        "article_title": article_title,
        "publication": publication_name,
        "lt_corrections_applied": lt_result.get("change_log", []) if lt_result else [],
        "lt_failed": lt_result.get("failed", False) if lt_result else True,
        "corrected_draft": corrected_draft,
        "api_call_log": api_call_log,
        "delta": delta,
        "section_1_consensus": consensus_flags,
        "section_2_fact_check": fact_check,
        "section_3_voice": openai_voice_result.get("data", {}).get("flags", []) if not openai_voice_result.get("failed") else [],
        "section_4_argument": mistral_argument_result.get("data", {}).get("flags", []) if not mistral_argument_result.get("failed") else [],
        "section_5_completeness": openai_completeness_result.get("data", {}).get("flags", []) if not openai_completeness_result.get("failed") else [],
        "section_6_red_team": red_team,
        "section_7_low_confidence": low_confidence,
        "section_8_additional": [],
        "model_failures": [
            name for name, result in [
                ("gemini_fact_check", gemini_result),
                ("openai_voice", openai_voice_result),
                ("mistral_argument", mistral_argument_result),
                ("openai_completeness", openai_completeness_result),
                ("mistral_red_team", mistral_redteam_result),
            ] if result.get("failed")
        ],
    }

    return report


def _compute_delta(current_draft, prior_report, current_consensus):
    import difflib

    prior_draft = prior_report.get("corrected_draft", "")
    prior_words = prior_draft.split()
    current_words = current_draft.split()

    if prior_words:
        diff = difflib.SequenceMatcher(None, prior_words, current_words)
        changed_words = sum(
            max(b2 - b1, d2 - d1)
            for tag, b1, b2, d1, d2 in diff.get_opcodes()
            if tag != "equal"
        )
        word_change_pct = round(changed_words / max(len(prior_words), 1) * 100, 1)
    else:
        word_change_pct = 100.0

    prior_consensus_passages = {
        _passage_key(f.get("passage", ""))
        for f in prior_report.get("section_1_consensus", [])
    }
    current_consensus_passages = {
        _passage_key(f.get("passage", ""))
        for f in current_consensus
    }

    resolved = prior_consensus_passages - current_consensus_passages
    new_flags = current_consensus_passages - prior_consensus_passages

    return {
        "word_change_pct": word_change_pct,
        "prior_consensus_count": len(prior_consensus_passages),
        "current_consensus_count": len(current_consensus_passages),
        "resolved_consensus_count": len(resolved),
        "new_consensus_count": len(new_flags),
    }


def rerun_recommended(delta, delta_config):
    if not delta:
        return False
    threshold = delta_config.get("word_change_threshold_pct", 15)
    if delta["word_change_pct"] > threshold:
        return True
    if delta["new_consensus_count"] > 0:
        return True
    return False
