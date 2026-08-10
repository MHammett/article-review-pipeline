"""
Render a consolidated review report (the dict produced by
``consolidation.build_report``) into a human-readable markdown document.

The saved JSON report holds everything the ensemble found, but nothing
renders it as prose a human can paste into a chat model or read directly —
only aggregate counts are printed to the console. This module fills that
gap, following the SECTION 1-8 structure documented in
``handoff_templates/review_report.md`` (section 9, citations, was added to
the pipeline after that template was written).
"""


def _kv_lines(d, exclude=()):
    """Render remaining key/value pairs of a flag dict as indented bullets."""
    lines = []
    for key, value in d.items():
        if key in exclude or value in (None, "", [], {}):
            continue
        label = key.replace("_", " ").capitalize()
        lines.append(f"  - {label}: {value}")
    return lines


def _render_section_1(consensus_flags):
    lines = ["## SECTION 1: Consensus Flags", ""]
    if not consensus_flags:
        lines.append("_No consensus flags._")
        return lines

    for i, entry in enumerate(consensus_flags, 1):
        passage = entry.get("passage", "")
        models = ", ".join(entry.get("models", []))
        weight = entry.get("weight_sum")
        lt = (
            " (LanguageTool also flagged)"
            if entry.get("languagetool_also_flagged")
            else ""
        )
        lines.append(f'### {i}. "{passage}"')
        lines.append(f"- Flagged by: {models} — weight {weight}{lt}")
        for flag in entry.get("flags", []):
            source = flag.get("source_model", "?")
            domain = flag.get("domain", "?")
            lines.append(f"  - **{source}:{domain}**")
            for kv in _kv_lines(
                flag,
                exclude=(
                    "domain",
                    "source_model",
                    "type",
                    "passage",
                    "passage_reference",
                ),
            ):
                lines.append(f"  {kv}")
        lines.append("")
    return lines


def _render_section_2(fact_check):
    lines = ["## SECTION 2: Factual Verification", ""]
    if not fact_check:
        lines.append("_No fact-check results._")
        return lines

    labels = {
        "confirmed": "Confirmed",
        "outdated": "Outdated",
        "contradicted": "Contradicted",
        "unverifiable": "Unverifiable",
        "primary_source_needed": "Primary source resolution required",
    }
    for key, label in labels.items():
        items = fact_check.get(key, [])
        if not items:
            continue
        lines.append(f"### {label}")
        for item in items:
            claim = item.get("claim", "")
            lines.append(f'- "{claim}"')
            for kv in _kv_lines(item, exclude=("claim",)):
                lines.append(kv)
        lines.append("")

    observations = fact_check.get("additional_observations", [])
    if observations:
        lines.append("### Additional observations (fact-check pass)")
        for obs in observations:
            passage = obs.get("passage") or obs.get("observation", "")
            lines.append(f'- "{passage}"')
            for kv in _kv_lines(obs, exclude=("passage",)):
                lines.append(kv)
        lines.append("")

    return lines


def _render_flags_section(title, flags, passage_key="passage"):
    lines = [f"## {title}", ""]
    if not flags:
        lines.append("_No flags._")
        return lines
    for flag in flags:
        passage = flag.get(passage_key, "")
        lines.append(f'- "{passage}"')
        for kv in _kv_lines(flag, exclude=(passage_key,)):
            lines.append(kv)
    lines.append("")
    return lines


def _render_red_team_entry(label, item):
    lines = [f'- **{label}**: "{item.get("passage", "")}"']
    for kv in _kv_lines(item, exclude=("passage",)):
        lines.append(kv)
    return lines


def _render_section_6(red_team):
    lines = ["## SECTION 6: Red Team Findings", ""]
    if not red_team:
        lines.append("_No red team results._")
        return lines

    rt_keys = (
        ("most_vulnerable_claim", "Most vulnerable claim"),
        ("highest_audience_risk", "Highest audience risk"),
        ("highest_credibility_risk", "Highest credibility risk"),
    )

    if "most_vulnerable_claim" in red_team or "highest_audience_risk" in red_team:
        # Single-source: flat dict with the three well-known keys.
        for key, label in rt_keys:
            item = red_team.get(key)
            if item:
                lines.extend(_render_red_team_entry(label, item))
        lines.append("")
    else:
        # Multi-source: keyed by model name.
        for model_name, data in red_team.items():
            weight = data.get("_weight")
            weight_str = f" (weight {weight})" if weight is not None else ""
            lines.append(f"### {model_name}{weight_str}")
            for key, label in rt_keys:
                item = data.get(key)
                if item:
                    lines.extend(_render_red_team_entry(label, item))
            lines.append("")

    return lines


def _render_section_7(low_confidence):
    lines = [
        "## SECTION 7: Low-Confidence Flags",
        "_For awareness only — dismiss unless something catches your attention._",
        "",
    ]
    if not low_confidence:
        lines.append("_None._")
        return lines
    for item in low_confidence:
        passage = item.get("passage") or item.get("passage_reference", "")
        source = item.get("source_model", "?")
        domain = item.get("domain", "?")
        lines.append(f'> "{passage}" — {source}:{domain}')
        for kv in _kv_lines(
            item, exclude=("passage", "passage_reference", "source_model", "domain")
        ):
            lines.append(f"> {kv.strip('- ')}")
    lines.append("")
    return lines


def _render_section_8(additional):
    lines = ["## SECTION 8: Additional Findings", ""]
    if not additional:
        lines.append("_None._")
        return lines
    for obs in additional:
        passage = obs.get("passage", "")
        category = obs.get("category", "?")
        source = obs.get("source_model", "?")
        domain = obs.get("source_domain", "?")
        lines.append(f'- [{category}] "{passage}" — flagged by {source}:{domain}')
        for kv in _kv_lines(
            obs, exclude=("passage", "category", "source_model", "source_domain")
        ):
            lines.append(kv)
    lines.append("")
    return lines


def _render_section_9(citations):
    lines = ["## SECTION 9: Citations", ""]
    if not citations:
        lines.append("_No citation resolution attempted._")
        return lines

    verified = [c for c in citations if c.get("verification") == "checksum"]
    pointer = [c for c in citations if c.get("verification") == "pointer"]
    unverifiable = [c for c in citations if c.get("verification") == "unverifiable"]
    unresolved = [c for c in citations if not c.get("resolved")]

    lines.append(
        f"{len(verified)} verified, {len(pointer)} pointer-only "
        f"(not independently verified), {len(unverifiable)} could not be verified, "
        f"{len(unresolved)} unresolved "
        f"— {len(citations)} claim(s) total"
    )
    lines.append("")

    changed = [c for c in verified if c.get("content_changed_since")]
    if changed:
        lines.append(f"### ⚠ Content changed since prior checksum ({len(changed)})")
        for c in changed:
            drift = c["content_changed_since"]
            when = f" on {drift['prior_date']}" if drift.get("prior_date") else ""
            lines.append(f'- "{c.get("claim", "")}"')
            lines.append(f"  - URL: {c.get('url', '')}")
            lines.append(
                f"  - Last matched in run {drift.get('prior_run')} of "
                f"'{drift.get('prior_article')}'{when} — content has since changed. "
                "Previously-verified claim may need re-checking."
            )
        lines.append("")

    if verified:
        lines.append("### Verified (checksummed against fetched content)")
        for c in verified:
            lines.append(f'- "{c.get("claim", "")}"')
            for kv in _kv_lines(
                c, exclude=("claim", "resolved", "content_changed_since")
            ):
                lines.append(kv)
        lines.append("")

    if pointer:
        lines.append(
            "### Pointer-only (topic-relevant source identified, NOT independently "
            "verified — confirm manually before citing)"
        )
        for c in pointer:
            lines.append(f'- "{c.get("claim", "")}"')
            for kv in _kv_lines(c, exclude=("claim", "resolved")):
                lines.append(kv)
        lines.append("")

    if unverifiable:
        lines.append(
            "### Could not be verified (source fetched, but its content could NOT "
            "be read or assessed — this is NOT a finding against the source)"
        )
        for c in unverifiable:
            lines.append(f'- "{c.get("claim", "")}"')
            for kv in _kv_lines(c, exclude=("claim", "resolved")):
                lines.append(kv)
        lines.append("")

    if unresolved:
        lines.append("### Unresolved")
        for c in unresolved:
            lines.append(f'- "{c.get("claim", "")}"')
            for kv in _kv_lines(c, exclude=("claim", "resolved")):
                lines.append(kv)
        lines.append("")

    return lines


def render_report_markdown(report):
    """Render a review report dict into a readable markdown document.

    ``report`` has the same shape saved to ``run_N_<timestamp>_report.json``
    by ``consolidation.build_report``.
    """
    lines = [
        "# CONSOLIDATED REVIEW REPORT",
        f"Generated: {report.get('generated', '')}",
        f"Pipeline run: {report.get('run_number', '')}",
        f"Article: {report.get('article_title', '')}",
        f"Publication: {report.get('publication', '')}",
        "",
    ]

    corrections = report.get("lt_corrections_applied", [])
    if report.get("lt_skipped"):
        lines.append("LanguageTool: skipped (no credentials configured)")
    elif report.get("lt_failed"):
        lines.append("LanguageTool: FAILED — draft not grammar-corrected")
    else:
        lines.append(f"LanguageTool corrections applied: {len(corrections)}")
    lines.append("")

    if report.get("model_failures"):
        lines.append(
            f"WARNING — failed model passes: {', '.join(report['model_failures'])}"
        )
        lines.append("")

    if report.get("truncated_results"):
        lines.append(
            "WARNING — truncated model responses (output-token ceiling hit; "
            "some findings were recovered, some were lost): "
            f"{', '.join(report['truncated_results'])}"
        )
        lines.append("")

    delta = report.get("delta")
    if delta:
        lines.append("## Delta From Prior Run")
        lines.append(f"- Word change: {delta.get('word_change_pct')}%")
        lines.append(
            f"- Resolved consensus flags: {delta.get('resolved_consensus_count')}/{delta.get('prior_consensus_count')}"
        )
        lines.append(f"- New consensus flags: {delta.get('new_consensus_count')}")
        if delta.get("claim_changed"):
            lines.append("- Primary claim: CHANGED since prior run")
        if delta.get("structure_changed"):
            lines.append("- Heading structure: CHANGED since prior run")
        lines.append("")

    lines.append("---")
    lines.append("")

    lines.extend(_render_section_1(report.get("section_1_consensus", [])))
    lines.extend(_render_section_2(report.get("section_2_fact_check", {})))
    lines.extend(
        _render_flags_section(
            "SECTION 3: Voice and AI-Speak", report.get("section_3_voice", [])
        )
    )
    lines.extend(
        _render_flags_section(
            "SECTION 4: Argument Integrity", report.get("section_4_argument", [])
        )
    )
    lines.extend(
        _render_flags_section(
            "SECTION 5: Completeness and Framing",
            report.get("section_5_completeness", []),
            passage_key="passage_reference",
        )
    )
    lines.extend(_render_section_6(report.get("section_6_red_team", {})))
    lines.extend(_render_section_7(report.get("section_7_low_confidence", [])))
    lines.extend(_render_section_8(report.get("section_8_additional", [])))
    lines.extend(_render_section_9(report.get("section_9_citations", [])))

    return "\n".join(lines).rstrip() + "\n"
