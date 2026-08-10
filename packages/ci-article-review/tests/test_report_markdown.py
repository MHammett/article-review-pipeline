"""Unit tests for report_markdown.render_report_markdown."""

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


class TestEmptyReport:
    def test_no_crash_on_all_empty_sections(self):
        md = render_report_markdown(_base_report())
        assert "SECTION 1" in md
        assert "SECTION 9" in md
