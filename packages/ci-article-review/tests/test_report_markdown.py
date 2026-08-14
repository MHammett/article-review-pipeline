"""Unit tests for report_markdown.render_report_markdown."""

from ci_article_review import report_markdown
from ci_article_review.report_markdown import render_report_markdown


def _base_report(**overrides):
    report = {
        "generated": "2026-08-09T00:00:00+00:00",
        "run_number": 1,
        "article_title": "Test Article",
        "publication": "test_pub",
        "lt_corrections_applied": [],
        "lt_failed": False,
        "lt_skipped": False,
        "model_failures": [],
        "delta": None,
        "section_1_consensus": [],
        "section_2_fact_check": {},
        "section_3_voice": [],
        "section_4_argument": [],
        "section_5_completeness": [],
        "section_6_red_team": {},
        "section_7_low_confidence": [],
        "section_8_additional": [],
        "section_9_citations": [],
    }
    report.update(overrides)
    return report


class TestHeader:
    def test_header_fields_present(self):
        md = render_report_markdown(_base_report())
        assert "Test Article" in md
        assert "test_pub" in md
        assert "Run" in md or "run" in md

    def test_lt_skipped_noted(self):
        md = render_report_markdown(_base_report(lt_skipped=True))
        assert "skipped" in md.lower()

    def test_delta_rendered(self):
        report = _base_report(
            delta={
                "word_change_pct": 12.5,
                "resolved_consensus_count": 1,
                "prior_consensus_count": 2,
                "new_consensus_count": 1,
                "claim_changed": True,
                "structure_changed": False,
            }
        )
        md = render_report_markdown(report)
        assert "12.5" in md
        assert "CHANGED" in md


class TestSection1Consensus:
    def test_passage_and_models_rendered(self):
        report = _base_report(
            section_1_consensus=[
                {
                    "passage": "Grande Reserve has approximately 2,200 planned homes",
                    "models": [
                        "mistral:argument_integrity",
                        "openai:argument_integrity",
                    ],
                    "weight_sum": 3.3,
                    "languagetool_also_flagged": False,
                    "flags": [
                        {
                            "passage": "Grande Reserve has approximately 2,200 planned homes",
                            "domain": "argument_integrity",
                            "source_model": "mistral",
                            "logical_problem": "Figure is asserted without a cited source.",
                            "steelman_considered": "Could be a well-known local figure.",
                            "why_it_survived": "No source is given anywhere in the piece.",
                        }
                    ],
                }
            ]
        )
        md = render_report_markdown(report)
        assert "Grande Reserve has approximately 2,200 planned homes" in md
        assert "mistral:argument_integrity" in md
        assert "3.3" in md
        assert "Figure is asserted without a cited source." in md
        assert "No source is given anywhere in the piece." in md


class TestSection2FactCheck:
    def test_outdated_and_contradicted_rendered(self):
        report = _base_report(
            section_2_fact_check={
                "confirmed": [],
                "outdated": [
                    {
                        "claim": "The population is 45,000.",
                        "current_value": "The 2025 estimate is 52,000.",
                        "source": "Census.gov",
                        "confidence": "high",
                    }
                ],
                "contradicted": [
                    {
                        "claim": "The bridge opened in 1990.",
                        "contradiction": "County records show 1992.",
                        "source": "County archives",
                        "confidence": "medium",
                    }
                ],
                "unverifiable": [],
                "primary_source_needed": [],
                "additional_observations": [],
            }
        )
        md = render_report_markdown(report)
        assert "The population is 45,000." in md
        assert "The 2025 estimate is 52,000." in md
        assert "The bridge opened in 1990." in md
        assert "County records show 1992." in md


class TestSection3Voice:
    def test_flag_rendered_with_rewrite(self):
        report = _base_report(
            section_3_voice=[
                {
                    "passage": "It remains to be seen how this will play out.",
                    "problem": "Vague significance gesturing with no specific claim.",
                    "suggested_rewrite": "Cut the sentence or state the actual prediction.",
                    "steelman_considered": "Could be intentional understatement.",
                    "source_model": "openai",
                }
            ]
        )
        md = render_report_markdown(report)
        assert "It remains to be seen how this will play out." in md
        assert "Vague significance gesturing with no specific claim." in md
        assert "Cut the sentence or state the actual prediction." in md
        assert "openai" in md


class TestSection4Argument:
    def test_flag_rendered(self):
        report = _base_report(
            section_4_argument=[
                {
                    "passage": "Therefore the plan will fail.",
                    "logical_problem": "Conclusion outruns the evidence presented.",
                    "steelman_considered": "Could rely on context established earlier.",
                    "why_it_survived": "No such context appears earlier in the piece.",
                    "source_model": "mistral",
                }
            ]
        )
        md = render_report_markdown(report)
        assert "Therefore the plan will fail." in md
        assert "Conclusion outruns the evidence presented." in md
        assert "mistral" in md


class TestSection5Completeness:
    def test_flag_rendered(self):
        report = _base_report(
            section_5_completeness=[
                {
                    "what_is_missing": "No mention of the county's competing proposal.",
                    "passage_reference": "The council approved the zoning change unanimously.",
                    "audience_affected": "Residents following the competing proposal.",
                    "source_model": "openai",
                }
            ]
        )
        md = render_report_markdown(report)
        assert "The council approved the zoning change unanimously." in md
        assert "No mention of the county's competing proposal." in md


class TestSection6RedTeam:
    def test_single_source_rendered(self):
        report = _base_report(
            section_6_red_team={
                "most_vulnerable_claim": {
                    "passage": "The project cost $4 million.",
                    "attack_vector": "Public records show a different figure.",
                    "supporting_evidence_for_attack": "County budget document.",
                },
                "highest_audience_risk": {
                    "passage": "Residents overwhelmingly support the plan.",
                    "risk": "No polling data is cited.",
                    "audience_segment": "Skeptical local readers.",
                },
                "highest_credibility_risk": {
                    "passage": "As an expert in this field, I can say...",
                    "risk": "Unverified expertise claim.",
                    "attack_vector": "Critic questions author's credentials.",
                },
            }
        )
        md = render_report_markdown(report)
        assert "The project cost $4 million." in md
        assert "Public records show a different figure." in md
        assert "Residents overwhelmingly support the plan." in md

    def test_multi_source_rendered(self):
        report = _base_report(
            section_6_red_team={
                "mistral": {
                    "_weight": 1.1,
                    "most_vulnerable_claim": {
                        "passage": "mistral finding",
                        "attack_vector": "a",
                        "supporting_evidence_for_attack": "b",
                    },
                },
                "grok": {
                    "_weight": 1.2,
                    "most_vulnerable_claim": {
                        "passage": "grok finding",
                        "attack_vector": "c",
                        "supporting_evidence_for_attack": "d",
                    },
                },
            }
        )
        md = render_report_markdown(report)
        assert "mistral finding" in md
        assert "grok finding" in md
        assert "mistral" in md
        assert "grok" in md


class TestSection7LowConfidence:
    def test_flagged_for_awareness_only(self):
        report = _base_report(
            section_7_low_confidence=[
                {
                    "passage": "This might be worth double-checking.",
                    "observation": "Did not survive the steelman filter.",
                    "source_model": "claude",
                    "domain": "argument_integrity",
                }
            ]
        )
        md = render_report_markdown(report)
        assert "for awareness only" in md.lower()
        assert "This might be worth double-checking." in md
        assert "Did not survive the steelman filter." in md


class TestSection8Additional:
    def test_observation_rendered(self):
        report = _base_report(
            section_8_additional=[
                {
                    "category": "fact_check",
                    "passage": "The county has 12 school districts.",
                    "observation": "Worth verifying against the state education dept.",
                    "confidence": "medium",
                    "source_model": "openai",
                    "source_domain": "voice_style",
                    "in_domain": False,
                }
            ]
        )
        md = render_report_markdown(report)
        assert "The county has 12 school districts." in md
        assert "Worth verifying against the state education dept." in md
        assert "fact_check" in md


class TestSection9Citations:
    def test_resolved_and_unresolved_separated(self):
        report = _base_report(
            section_9_citations=[
                {
                    "claim": "The bridge cost $4 million.",
                    "resolved": True,
                    "source_name": "County Budget",
                    "url": "https://example.gov/budget.pdf",
                    "verification": "checksum",
                    "wayback": {"archived": True, "snapshot_age_days": 10},
                },
                {
                    "claim": "The plan was approved in 2021.",
                    "resolved": False,
                    "note": "No configured source adapter could resolve this claim",
                },
            ]
        )
        md = render_report_markdown(report)
        assert "### Verified" in md
        assert "### Unresolved" in md
        assert "The bridge cost $4 million." in md
        assert "https://example.gov/budget.pdf" in md
        assert "The plan was approved in 2021." in md
        assert "No configured source adapter could resolve this claim" in md

    def test_verified_and_pointer_reported_distinctly(self):
        """A checksum-verified fetch and a pointer-only match must not be
        collapsed into one "resolved" bucket — see the bug this guards against:
        both used to render under the same "### Resolved" heading with no way
        to tell a verified claim from an unverified topic pointer.
        """
        report = _base_report(
            section_9_citations=[
                {
                    "claim": "The bridge cost $4 million.",
                    "resolved": True,
                    "source_name": "County Budget",
                    "url": "https://example.gov/budget.pdf",
                    "verification": "checksum",
                    "wayback": {"archived": True},
                },
                {
                    "claim": "PJM's latest capacity auction cleared at a record price.",
                    "resolved": True,
                    "source_name": "pjm",
                    "url": "https://www.pjm.com/markets-and-operations/rpm",
                    "verification": "pointer",
                    "wayback": {"archived": False},
                },
            ]
        )
        md = render_report_markdown(report)
        assert "### Verified" in md
        assert "### Pointer-only" in md
        assert "not independently verified" in md.lower()

        verified_section = md.split("### Verified")[1].split("### Pointer-only")[0]
        pointer_section = md.split("### Pointer-only")[1]
        assert "The bridge cost $4 million." in verified_section
        assert "capacity auction" not in verified_section.lower()
        assert "capacity auction" in pointer_section.lower()
        assert "The bridge cost $4 million." not in pointer_section

    def test_unverifiable_rendered_separately_from_mismatch(self):
        """ "We could not read the source" and "the source does not support the
        claim" are different findings. A citation we never managed to read must
        never appear as evidence against its source, and must not silently
        vanish from the report either.
        """
        report = _base_report(
            section_9_citations=[
                {
                    "claim": "ICNIRP sets a 200 microtesla reference level.",
                    "resolved": True,
                    "url": "https://www.icnirp.org/gdl.pdf",
                    "verification": "unverifiable",
                    "content_kind": "pdf",
                    "note": "no text could be extracted from the PDF",
                    "wayback": {"archived": True},
                },
                {
                    "claim": "The plan was approved in 2021.",
                    "resolved": False,
                    "verification": "content_mismatch",
                    "note": "content verification found it does not support",
                },
            ]
        )
        md = render_report_markdown(report)

        assert "### Could not be verified" in md
        assert "could not be verified" in md.lower()

        unverifiable_section = md.split("### Could not be verified")[1].split("###")[0]
        assert "ICNIRP" in unverifiable_section
        assert "does not support" not in unverifiable_section
        # The mismatch stays in its own bucket.
        assert "The plan was approved in 2021." not in unverifiable_section

    def test_unverifiable_counted_in_summary_line(self):
        report = _base_report(
            section_9_citations=[
                {
                    "claim": "c",
                    "resolved": True,
                    "verification": "unverifiable",
                    "note": "n",
                }
            ]
        )
        md = render_report_markdown(report)

        assert "1 could not be verified" in md

    def test_content_drift_called_out_above_the_tiers(self):
        """A source that changed since it was last checksummed gets its own
        block — buried in the verified entry's key/value bullets it would read
        as just another field."""
        report = _base_report(
            section_9_citations=[
                {
                    "claim": "The bridge cost $4 million.",
                    "resolved": True,
                    "source_name": "County Budget",
                    "url": "https://example.gov/budget.pdf",
                    "verification": "checksum",
                    "wayback": {"archived": True},
                    "content_changed_since": {
                        "prior_checksum": "abc123",
                        "prior_run": 4,
                        "prior_article": "prior-article",
                        "prior_date": "2026-01-01T00:00:00",
                        "note": "changed",
                    },
                },
            ]
        )
        md = render_report_markdown(report)
        assert "### ⚠ Content changed since prior checksum (1)" in md

        drift_section = md.split("### ⚠ Content changed")[1].split("### Verified")[0]
        assert "https://example.gov/budget.pdf" in drift_section
        assert "run 4 of 'prior-article' on 2026-01-01T00:00:00" in drift_section
        assert "may need re-checking" in drift_section
        # The entry still renders in its tier; it is not moved out of Verified.
        assert "The bridge cost $4 million." in md.split("### Verified")[1]

    def test_no_drift_block_when_nothing_changed(self):
        report = _base_report(
            section_9_citations=[
                {
                    "claim": "The bridge cost $4 million.",
                    "resolved": True,
                    "url": "https://example.gov/budget.pdf",
                    "verification": "checksum",
                },
            ]
        )
        assert "Content changed since prior checksum" not in render_report_markdown(
            report
        )


def _field(label, value="", limit=None, rationale="", default_note="", **extra):
    field = {
        "label": label,
        "value": value,
        "rationale": rationale,
        "chars": len(value) if value else None,
        "limit": limit if value else None,
        "over_limit": bool(value and limit is not None and len(value) > limit),
        "default_note": "" if value else default_note,
    }
    field.update(extra)
    return field


class TestSeoSuggestions:
    _OK = {
        "status": "ok",
        "model": "mistral-small-latest",
        "keyword_candidates": [
            {"keyword": "interconnection queue", "rationale": "what officials search"},
            {"keyword": "grid capacity", "rationale": "broader intent"},
        ],
        "fields": {
            "meta_description": _field(
                "Meta description",
                value="Queues, not generation, decide the timeline.",
                limit=155,
            ),
            "og_title": _field(
                "OG title",
                default_note="The article title is used as-is.",
            ),
            "og_description": _field(
                "OG description",
                default_note="The meta description is used.",
            ),
            "schema_type": _field(
                "Schema type",
                value="BlogPosting",
                rationale="commentary in a personal voice",
                recognized=True,
                configured_default="BlogPosting",
                differs_from_default=False,
            ),
        },
    }

    def _render(self, suggestions):
        return render_report_markdown(
            _base_report(pre_analysis={"seo": {"suggestions": suggestions}})
        )

    def _with_field(self, name, field):
        return {**self._OK, "fields": {**self._OK["fields"], name: field}}

    def test_candidates_and_description_rendered(self):
        md = self._render(self._OK)
        assert "## SEO Suggestions" in md
        assert "interconnection queue" in md
        assert "what officials search" in md
        assert "Queues, not generation, decide the timeline." in md
        assert "44/155 chars" in md

    def test_states_that_nothing_was_applied(self):
        # This section is pasted into a chat model that is about to revise the
        # article — it must not read as a decision already made.
        md = self._render(self._OK)
        assert "not decided" in md
        assert "Do not select a focus keyword" in md

    def test_over_limit_description_is_marked(self):
        md = self._render(
            self._with_field(
                "meta_description",
                _field("Meta description", value="x" * 200, limit=155),
            )
        )
        assert "over the limit" in md

    def test_every_metadata_field_is_accounted_for(self):
        # A field with no proposal still says which default the push applies —
        # silence would read as "not considered".
        md = self._render(self._OK)
        assert "Meta description" in md
        assert "OG title" in md and "The article title is used as-is." in md
        assert "OG description" in md and "The meta description is used." in md
        assert "Schema type" in md

    def test_og_title_rendered_when_proposed(self):
        md = self._render(
            self._with_field(
                "og_title", _field("OG title", value="A Shorter Title", limit=60)
            )
        )
        assert "OG title" in md
        assert "A Shorter Title" in md
        assert "15/60 chars" in md

    def test_og_description_rendered_when_proposed(self):
        md = self._render(
            self._with_field(
                "og_description",
                _field("OG description", value="A punchier social hook.", limit=155),
            )
        )
        assert "A punchier social hook." in md

    def test_schema_type_rationale_and_default_comparison(self):
        md = self._render(
            self._with_field(
                "schema_type",
                _field(
                    "Schema type",
                    value="NewsArticle",
                    rationale="reporting tied to a pending vote",
                    recognized=True,
                    configured_default="BlogPosting",
                    differs_from_default=True,
                ),
            )
        )
        assert "NewsArticle" in md
        assert "reporting tied to a pending vote" in md
        assert "Differs from the configured default" in md
        assert "BlogPosting" in md

    def test_unrecognized_schema_type_is_flagged(self):
        md = self._render(
            self._with_field(
                "schema_type",
                _field(
                    "Schema type",
                    value="TechArticle",
                    recognized=False,
                    configured_default="BlogPosting",
                    differs_from_default=True,
                ),
            )
        )
        assert "TechArticle" in md
        assert "confirm" in md.lower()

    def test_unavailable_reason_is_shown(self):
        md = self._render({"status": "failed", "reason": "call failed: 503"})
        assert "## SEO Suggestions" in md
        assert "call failed: 503" in md

    def test_absent_when_no_suggestion_pass_ran(self):
        md = render_report_markdown(_base_report(pre_analysis={"seo": {}}))
        assert "SEO Suggestions" not in md

    def test_absent_for_reports_predating_the_pass(self):
        md = render_report_markdown(_base_report())
        assert "SEO Suggestions" not in md


class TestSeoKeywordUsage:
    def _render(self, usage):
        suggestions = {
            "status": "ok",
            "keyword_candidates": [
                {"keyword": "interconnection queue", "rationale": "why", "usage": usage}
            ],
            "fields": {},
        }
        return render_report_markdown(
            _base_report(pre_analysis={"seo": {"suggestions": suggestions}})
        )

    def test_unused_phrase_is_called_out(self):
        md = self._render(
            {"in_title": False, "in_headings": [], "in_opening": False, "body_count": 0}
        )
        assert "never uses this phrase" in md

    def test_used_phrase_reports_where(self):
        md = self._render(
            {
                "in_title": True,
                "in_headings": ["## A heading"],
                "in_opening": True,
                "body_count": 6,
            }
        )
        assert "Appears 6x" in md
        assert "the title" in md
        assert "the opening" in md
        assert "1 heading(s)" in md

    def test_unscanned_candidate_renders_without_a_usage_line(self):
        md = self._render(None)
        assert "interconnection queue" in md
        assert "Appears" not in md


class TestSeoContentReview:
    _FINDING = {
        "type": "heading",
        "target": "The Bigger Picture",
        "problem": "Could sit above any section of any article.",
        "suggestion": "How a queue position becomes a five-year wait",
    }

    def _render(self, content_review):
        return render_report_markdown(
            _base_report(pre_analysis={"seo": {"content_review": content_review}})
        )

    def test_findings_rendered_with_suggestions(self):
        md = self._render({"status": "ok", "findings": [self._FINDING]})
        assert "## SEO Structure Review" in md
        assert "The Bigger Picture" in md
        assert "Could sit above any section of any article." in md
        assert "How a queue position becomes a five-year wait" in md

    def test_clean_article_says_so_rather_than_going_blank(self):
        md = self._render({"status": "ok", "findings": []})
        assert "## SEO Structure Review" in md
        assert "Nothing flagged" in md

    def test_scope_is_stated_so_it_is_not_read_as_a_full_review(self):
        md = self._render({"status": "ok", "findings": [self._FINDING]})
        assert "Structure only" in md

    def test_unavailable_reason_is_shown(self):
        md = self._render({"status": "failed", "reason": "call failed: 503"})
        assert "call failed: 503" in md

    def test_absent_when_the_pass_did_not_run(self):
        assert "SEO Structure Review" not in render_report_markdown(_base_report())


class TestEmptyReport:
    def test_no_crash_on_all_empty_sections(self):
        md = render_report_markdown(_base_report())
        assert "SECTION 1" in md
        assert "SECTION 9" in md


class TestCitationsPairLiveAndArchiveLinks:
    """Every citation should carry both links, so it survives its source.

    The snapshot URL was collected on every run and rendered nowhere — the
    pairing existed in the report JSON and never in anything a human read.
    """

    def _cit(self, **wayback):
        return {
            "claim": "A claim",
            "url": "https://example.org/page",
            "resolved": True,
            "verification": "checksum",
            "wayback": wayback,
        }

    def _text(self, citation):
        return "\n".join(report_markdown._render_section_9([citation]))

    def test_an_archived_citation_shows_both_links(self):
        out = self._text(
            self._cit(archived=True, snapshot_url="https://web.archive.org/web/1/x")
        )
        assert "Live: https://example.org/page" in out
        assert "Archive: https://web.archive.org/web/1/x" in out

    def test_a_paste_ready_pairing_is_offered(self):
        """The point is a citation the author can put in the article as-is."""
        out = self._text(
            self._cit(archived=True, snapshot_url="https://web.archive.org/web/1/x")
        )
        assert (
            "Cite both: https://example.org/page (archived: https://web.archive.org/web/1/x)"
            in out
        )

    def test_a_stale_snapshot_is_marked_where_it_is_read(self):
        out = self._text(
            self._cit(
                archived=True,
                snapshot_url="https://web.archive.org/web/1/x",
                snapshot_stale=True,
            )
        )
        assert "STALE" in out

    def test_a_just_submitted_archive_says_so_rather_than_going_quiet(self):
        out = self._text(self._cit(archived=False, submitted=True))
        assert "submitted to the Wayback Machine this run" in out
        assert "Cite both" not in out

    def test_an_unarchived_citation_says_it_is_undurable(self):
        """Silence would read as "fine"; it is the case that needs action."""
        out = self._text(self._cit(archived=False))
        assert "only as durable as the live URL" in out

    def test_a_citation_with_no_url_renders_no_pairing(self):
        citation = {"claim": "c", "resolved": True, "verification": "checksum"}
        assert report_markdown._render_archive_pair(citation) == []
