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

#: Order the SEO METADATA fields render in, matching publication.md's block.
#: Duplicated from ``analysis.seo_suggest.FIELD_ORDER`` rather than imported so
#: this module stays a dependency-free renderer over a plain dict — importing
#: the suggestion module would pull the provider adapters in behind it. A test
#: asserts the two stay in step.
_SEO_FIELD_ORDER = ("meta_description", "og_title", "og_description", "schema_type")


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


def _citation_pair(citation):
    """Return the live URL and its archive URL, for a citation the author can paste.

    A citation that names only a live URL is durable until the page moves,
    changes, or 403s — which link-check runs show happening constantly. Pairing
    every live link with its Wayback snapshot is what makes the citation survive
    the source. The pipeline has always *collected* the snapshot URL; it was
    never rendered anywhere a human would see it, so the pairing existed in the
    data and nowhere in the output.

    Returns ``(live_url, archive_url_or_None)``.
    """
    wayback = citation.get("wayback") or {}
    return citation.get("url", ""), wayback.get("snapshot_url")


def _render_archive_pair(citation, indent="  "):
    """Lines pairing a citation's live URL with its archive copy.

    Says which of the four states applies rather than silently omitting the
    archive line, because "no snapshot yet" and "never submitted" need different
    follow-up from the author.

    The fourth state is the one worth being careful about. ``archived: None``
    means the lookup never completed — the circuit breaker above tripped, or the
    request failed — and it is NOT the same as "there is no snapshot". Reporting
    it as "none" would assert something this run never established, and the
    breaker makes that the common case rather than a rare one: once it trips,
    every remaining citation carries a null.
    """
    live, archive = _citation_pair(citation)
    if not live:
        return []
    out = [f"{indent}- Live: {live}"]
    wayback = citation.get("wayback") or {}
    if archive:
        stale = (
            " (STALE — re-archive before relying on it)"
            if wayback.get("snapshot_stale")
            else ""
        )
        out.append(f"{indent}- Archive: {archive}{stale}")
        out.append(f"{indent}- Cite both: {live} (archived: {archive})")
    elif wayback.get("submitted"):
        out.append(
            f"{indent}- Archive: submitted to the Wayback Machine this run; the "
            f"snapshot URL appears on the next run once archive.org has captured it."
        )
    elif wayback.get("archived") is None:
        out.append(
            f"{indent}- Archive: NOT CHECKED — the archive.org lookup did not "
            f"complete this run. This says nothing about whether the page is "
            f"archived; re-run to find out."
        )
    else:
        out.append(
            f"{indent}- Archive: none. This citation is only as durable as the "
            f"live URL — re-run once archiving succeeds, or archive it by hand."
        )
    return out


#: Disposition buckets, strongest retrieval first. The order is the order the
#: reader meets them: "we fetched and read the document" before "we never looked
#: it up", so the section opens on its best evidence and degrades honestly.
#:
#: Keyed by ``verification`` tier, with the untiered entries split in two. That
#: split matters: an entry with no tier but a URL was a real fetch that was
#: refused (403, 404, DNS), which is a different fact about the claim than never
#: having had a URL at all — and it is usually actionable, because a publisher
#: that refuses an automated fetch will often serve the same page to a person.
_DISPOSITIONS = (
    ("checksum", "Read, and supports the claim"),
    ("content_mismatch", "Read, and does NOT support the claim"),
    ("unverifiable", "Fetched, but could not be read"),
    ("fetch_failed", "Source URL identified, but the fetch was refused"),
    ("pointer", "Pointer only — nothing retrieved"),
    ("no_source", "No source identified"),
)

#: The two dispositions where a document was genuinely retrieved *and* its text
#: read by the relevance check. "Checked" means these and only these — the
#: verdict then splits them. Conflating "checked" with "supports" is the same
#: mistake this section exists to stop a reader making, one level up.
_READ_DISPOSITIONS = ("checksum", "content_mismatch")


def _disposition(citation):
    """Which ``_DISPOSITIONS`` bucket a citation belongs in.

    Anything that never reached a verification tier is one of the two "nothing
    was read" buckets, regardless of ``resolved``: ``fetch_failed`` when a URL
    was identified and the fetch did not succeed, ``no_source`` when there was
    no URL to try.
    """
    tier = citation.get("verification")
    if tier in {k for k, _ in _DISPOSITIONS}:
        return tier
    return "fetch_failed" if citation.get("url") else "no_source"


def _render_section_9(citations):
    # The heading carries the framing deliberately. "Citations" alone reads as a
    # list of sources backing the article, which invites more confidence than
    # the tiers below have earned — in a real run 18 of 144 claims had a document
    # fetched and read, and the rest rested on a model asserting a source
    # exists. "SECTION 9" and "Citations" both stay greppable: the paste-based
    # revision loop in handoff_templates/revise_after_review_prompt.md refers to
    # this section by name.
    lines = ["## SECTION 9: Citations — what was actually checked", ""]
    if not citations:
        lines.append("_No citation resolution attempted._")
        return lines

    grouped = {key: [] for key, _ in _DISPOSITIONS}
    for c in citations:
        grouped[_disposition(c)].append(c)

    verified = grouped["checksum"]
    mismatch = grouped["content_mismatch"]
    unverifiable = grouped["unverifiable"]
    fetch_failed = grouped["fetch_failed"]
    pointer = grouped["pointer"]
    no_source = grouped["no_source"]
    total = len(citations)
    checked = len(verified) + len(mismatch)

    # Lead with the fraction, not a tier breakdown. The breakdown was accurate
    # but made the reader do arithmetic across four numbers to learn the one
    # thing that governs how much of this section to trust.
    #
    # "Checked" counts both read dispositions, not just the supporting one. A
    # claim whose source was fetched, read, and found not to back it was checked
    # — the check returned "no". Reporting only the confirmations as "checked"
    # would repeat, in the summary line, exactly the conflation between "a model
    # says so" and "a document was read" that the tiers below exist to separate.
    lines.append(
        f"**{checked} of {total} claim(s) ({round(100 * checked / total)}%) were "
        f"checked against a document the pipeline fetched and read** — "
        f"{len(verified)} where the document supported the claim, "
        f"{len(mismatch)} where it did not."
    )
    lines.append("")
    lines.append(
        f"For the other {total - checked}, no document was read. What stands "
        "behind those claims is a model asserting that a source exists and "
        "supports it — recalled from training data or read off a search result. "
        "That is a research lead, not a verification. The table says which is "
        "which; every claim is in exactly one row."
    )
    lines.append("")
    lines.append("| What happened | Claims |")
    lines.append("| --- | ---: |")
    for key, label in _DISPOSITIONS:
        lines.append(f"| {label} | {len(grouped[key])} |")
    lines.append("")

    # Relate the fact-check pass's own verdict to whether anything was retrieved.
    # These are independent — one is model judgment, the other is retrieval — and
    # a reader who sees "Bucket: confirmed" beside a claim with no URL reasonably
    # reads it as corroboration. In the run that motivated this, 85 claims came
    # back "confirmed"; 9 were read and supported, 9 were read and were not, and
    # for the remaining 67 no document was read at all.
    confirmed = [c for c in citations if c.get("fact_check_bucket") == "confirmed"]
    if confirmed:
        supported = sum(1 for c in confirmed if _disposition(c) == "checksum")
        refuted = sum(1 for c in confirmed if _disposition(c) == "content_mismatch")
        unread = len(confirmed) - supported - refuted
        lines.append(
            f"> **The fact-check pass called {len(confirmed)} of these claims "
            f'"confirmed." Of those, {supported} had a document fetched and read '
            f"here that supports the claim, {refuted} had one that does not, and "
            f"for {unread} no document was read at all.** A `confirmed` bucket is "
            "that pass's judgment about the claim, not a retrieval result. Where "
            "the two disagree, this section is the one that opened the document."
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
        # One-line gloss per tier. CITATIONS.md calibrates these carefully, but
        # the reader meets them here first, and "Verified" alone invites more
        # trust than the tier earns (audit finding 18).
        lines.append(
            f"### Read, and supports the claim ({len(verified)}) — checksummed "
            "against fetched content"
        )
        lines.append(
            "_Source fetched, checksummed, and a model confirmed the extracted "
            "text supports the claim — with a quote checked against the page. "
            "The strongest tier; still one cheap model call, not a human._"
        )
        lines.append("")
        for c in verified:
            lines.append(f'- "{c.get("claim", "")}"')
            lines.extend(_render_archive_pair(c))
            for kv in _kv_lines(
                c, exclude=("claim", "resolved", "content_changed_since")
            ):
                lines.append(kv)
        lines.append("")

    if mismatch:
        # These used to render inside "Unresolved", indistinguishable from
        # claims nothing was ever fetched for. They are the opposite: the only
        # rows in the section where a document was retrieved, read, and found
        # not to back the claim it was cited for. README.md calls them among the
        # most actionable findings a run produces, so they get their own block,
        # directly under the confirmed ones.
        lines.append(
            f"### Read, and does NOT support the claim ({len(mismatch)}) — check these"
        )
        lines.append(
            "_The source fetched and read cleanly; the relevance check came back "
            "saying it does not back this specific claim. Read the verdict before "
            "reacting to it, because the two kinds mean different things._"
        )
        lines.append("")
        # "contradicts" means the document says otherwise and the draft may be
        # wrong. "not_addressed"/"inconclusive" much more often means the URL was
        # wrong or the extraction missed the relevant part — a citation problem,
        # not a factual one. Collapsing them would overstate the first and bury
        # the second.
        contradicted = [
            c for c in mismatch if c.get("relevance_verdict") == "contradicts"
        ]
        if contradicted:
            lines.append(
                f"⚠ {len(contradicted)} of these came back `contradicts` — the "
                "source says something that conflicts with the claim. Treat those "
                "as a possible factual error, not a citation error."
            )
        else:
            lines.append(
                "None came back `contradicts`. Every entry here is "
                "`not_addressed` or `inconclusive`: the page did not cover the "
                "claim. That usually means the wrong URL was checked, or the "
                "relevant part of the page did not extract — verify the source is "
                "the one intended before treating it as a problem with the claim."
            )
        lines.append("")
        for c in mismatch:
            lines.append(f'- "{c.get("claim", "")}"')
            for kv in _kv_lines(c, exclude=("claim", "resolved")):
                lines.append(kv)
        lines.append("")

    if pointer:
        lines.append(
            f"### Pointer only ({len(pointer)}) — topic-relevant source "
            "identified, NOT independently verified (confirm manually before citing)"
        )
        lines.append(
            "_A keyword match pointed at a portal that is probably about the right "
            "topic. Nothing was retrieved or confirmed. Treat as a research lead._"
        )
        lines.append("")
        for c in pointer:
            lines.append(f'- "{c.get("claim", "")}"')
            lines.extend(_render_archive_pair(c))
            for kv in _kv_lines(c, exclude=("claim", "resolved")):
                lines.append(kv)
        lines.append("")

    if unverifiable:
        lines.append(
            f"### Fetched, but could not be read ({len(unverifiable)}) — its "
            "content could NOT be read or assessed (this is NOT a finding "
            "against the source)"
        )
        lines.append(
            "_A scanned PDF, a JavaScript-rendered page, a bot wall, or a "
            "relevance check that could not run. Treat exactly like pointer-only. "
            "The one thing it never means is that the source failed to back you._"
        )
        lines.append("")
        for c in unverifiable:
            lines.append(f'- "{c.get("claim", "")}"')
            for kv in _kv_lines(c, exclude=("claim", "resolved")):
                lines.append(kv)
        lines.append("")

    if fetch_failed:
        # Split out of the old "Unresolved" pile because it is the one bucket in
        # the section a reader can usually clear by hand: the exact document is
        # named, and a publisher that refuses an automated fetch (403) will often
        # serve the same page to a person in a browser. Lumping it in with claims
        # that never had a URL hid that.
        lines.append(
            f"### Source URL identified, but the fetch was refused "
            f"({len(fetch_failed)}) — worth opening by hand"
        )
        lines.append(
            "_A specific URL was named for these claims and the fetch did not "
            "succeed: refused (403), missing (404), or unreachable. A 403 is a "
            "statement about automated access, not about the document — these are "
            "often readable in a browser, and academic publishers in particular "
            "refuse every automated tier. Nothing here is evidence either way._"
        )
        lines.append("")
        for c in fetch_failed:
            lines.append(f'- "{c.get("claim", "")}"')
            for kv in _kv_lines(c, exclude=("claim", "resolved")):
                lines.append(kv)
        lines.append("")

    if no_source:
        # Previously "Unresolved", which quietly also held every content mismatch
        # and every refused fetch. Naming it for what happened keeps the bucket
        # honest: no URL was ever found, so nothing here is evidence either way.
        lines.append(f"### No source identified ({len(no_source)})")
        lines.append(
            "_No URL was found for these claims, so nothing was fetched and "
            "nothing was checked. This is not evidence against the claims — it is "
            "the absence of evidence about them. Usually the largest group, and "
            "the one most worth reading as "
            '"still to do" rather than as a result._'
        )
        lines.append("")
        for c in no_source:
            lines.append(f'- "{c.get("claim", "")}"')
            for kv in _kv_lines(c, exclude=("claim", "resolved")):
                lines.append(kv)
        lines.append("")

    return lines


def _render_seo_suggestions(pre_analysis):
    """Render the SEO suggestion block, if the pass produced one.

    This section exists to reach the chat revision round-trip (see
    ``handoff_templates/revise_after_review_prompt.md``), so it is written for
    a model as much as for a person — hence the explicit instruction not to
    treat any of it as decided. Keyword choice is the author's call, and a
    revision pass that quietly picks one would be making it for them.

    Returns [] when no suggestion pass ran, so reports predating this section
    (and runs with the pass disabled) render exactly as they did before.
    """
    suggestions = (pre_analysis or {}).get("seo", {}).get("suggestions")
    if not suggestions:
        return []

    lines = ["## SEO Suggestions", ""]
    if suggestions.get("status") != "ok":
        lines.append(
            f"_Not available this run: {suggestions.get('reason', 'unknown reason')}._"
        )
        lines.append("")
        return lines

    lines.append(
        "_Proposed, not decided. Nothing here has been written to any config, "
        "handoff, or WordPress metadata. Do not select a focus keyword on the "
        "author's behalf — that is a strategic choice about what to rank for._"
    )
    lines.append("")

    candidates = suggestions.get("keyword_candidates") or []
    if candidates:
        lines.append("### Focus keyword candidates")
        for c in candidates:
            rationale = f" — {c['rationale']}" if c.get("rationale") else ""
            lines.append(f"- **{c['keyword']}**{rationale}")
            usage = _keyword_usage_line(c.get("usage"))
            if usage:
                lines.append(f"  - {usage}")
        lines.append("")

    fields = suggestions.get("fields") or {}
    if fields:
        lines.append("### SEO METADATA fields")
        lines.append("")
        for name in _SEO_FIELD_ORDER:
            field = fields.get(name)
            if field:
                lines.extend(_render_seo_field(field))

    return lines


def _keyword_usage_line(usage):
    """Where a candidate phrase actually appears in the article.

    A mechanical scan, not a judgement — and the reason keyword candidates
    surface at draft stage at all. A phrase the article never uses is the
    finding a revision pass most needs to see.
    """
    if not usage:
        return ""
    if not usage.get("body_count"):
        return (
            "**The article never uses this phrase.** Either work it in where it "
            "fits naturally, or pick a candidate the piece already speaks to."
        )

    where = []
    if usage.get("in_title"):
        where.append("the title")
    if usage.get("in_opening"):
        where.append("the opening")
    headings = usage.get("in_headings") or []
    if headings:
        where.append(f"{len(headings)} heading(s)")
    placement = ", ".join(where) if where else "the body only"
    return f"Appears {usage['body_count']}x — in {placement}."


def _render_seo_content_review(pre_analysis):
    """Render the structural findings from the search-reader review pass."""
    content_review = (pre_analysis or {}).get("seo", {}).get("content_review")
    if not content_review:
        return []

    lines = ["## SEO Structure Review", ""]
    if content_review.get("status") != "ok":
        lines.append(
            f"_Not available this run: "
            f"{content_review.get('reason', 'unknown reason')}._"
        )
        lines.append("")
        return lines

    findings = content_review.get("findings") or []
    if not findings:
        lines.append(
            "_Nothing flagged — headings, opening, and title all read as "
            "delivering what a search reader arrived for._"
        )
        lines.append("")
        return lines

    lines.append(
        "_How the article reads to someone who just arrived from a search "
        "result. Structure only — the review sections above cover argument, "
        "completeness, and voice._"
    )
    lines.append("")
    for finding in findings:
        target = f': "{finding["target"]}"' if finding.get("target") else ""
        lines.append(f"- **{finding['type']}**{target}")
        lines.append(f"  - {finding['problem']}")
        if finding.get("suggestion"):
            lines.append(f"  - Suggested: {finding['suggestion']}")
    lines.append("")
    return lines


def _render_seo_field(field):
    """One row of the SEO METADATA table of outcomes.

    Fields with no proposed value still render. Two of them (OG title, OG
    description) have defaults the WordPress push applies on its own, and
    naming the default that would take effect is more use to the author than
    omitting the field and leaving them to wonder whether it was considered.
    """
    label = field.get("label", "")
    if not field.get("value"):
        return [f"**{label}:** {field.get('default_note', '_not proposed_')}", ""]

    measured = (
        f" ({field['chars']}/{field['limit']} chars)"
        if field.get("limit") is not None
        else ""
    )
    over = " — **over the limit, trim before use**" if field.get("over_limit") else ""
    lines = [f"**{label}**{measured}{over}", ""]

    if field.get("recognized") is False:
        lines.append(
            "_Not one of the types this publication's template lists — confirm "
            "Rank Math accepts it before using._"
        )
        lines.append("")
    elif field.get("differs_from_default"):
        lines.append(
            f"_Differs from the configured default "
            f"(`{field['configured_default']}`), which is what the push would "
            f"set if this field is left blank._"
        )
        lines.append("")

    lines.append(f"> {field['value']}")
    if field.get("rationale"):
        lines.append("")
        lines.append(f"_{field['rationale']}_")
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
        compared = (delta.get("compared_against") or {}).get("report")
        if compared:
            lines.append(f"- Compared against: `{compared}`")
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
    lines.extend(_render_seo_suggestions(report.get("pre_analysis", {})))
    lines.extend(_render_seo_content_review(report.get("pre_analysis", {})))

    return "\n".join(lines).rstrip() + "\n"
