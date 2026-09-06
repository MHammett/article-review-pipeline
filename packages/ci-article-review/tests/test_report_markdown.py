"""Unit tests for report_markdown.render_report_markdown."""

import re

from ci_article_review import report_markdown
from ci_article_review.adapters.citation import wayback
from ci_article_review.consolidation import _DOMAIN_SECTIONS
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


def _mismatch_citation(**overrides):
    """A content-mismatch citation shaped like the ones the resolver emits.

    Modelled on the Honda run's single refutation: a grounded-model citation
    that fetched and read cleanly, came back ``not_addressed``, and carries a
    ``note`` the resolver builds by restating the verdict and the reason. The
    URLs are long on purpose — the opaque redirect ones in that run were 271
    characters each, and three of them were what buried the actionable line.
    """
    citation = {
        "claim": "the chip advanced to exactly January 1, 2003.",
        "url": "https://example.test/redirect/" + "A" * 200,
        "resolved": True,
        "verification": "content_mismatch",
        "relevance_verdict": "not_addressed",
        "relevance_reason": "The page discusses 1998 or 2002, not 2003.",
        "note": (
            "Source URL loaded, and its extracted article text was read and "
            "checked, but content verification found it does not support this "
            "specific claim (not_addressed): The page discusses 1998 or 2002, "
            "not 2003."
        ),
        "content_summary": "SUMMARY " * 60,
        "checksum": "451bb861017fe7c4446b7dc274a2d6ce",
        "alternates_checked": [
            "https://example.test/alt-one/" + "B" * 200,
            "https://example.test/alt-two/" + "C" * 200,
        ],
        "wayback": {
            "archived": True,
            "snapshot_url": "https://web.archive.test/x",
            "snapshot_age_days": 10,
        },
    }
    citation.update(overrides)
    return citation


class TestFailedModelPassesAreExplained:
    """A failed pass has to say what happened and what it cost the report.

    The whole of it used to be "WARNING — failed model passes:
    openai:fact_check". That names the casualty but not the cause ("Response
    ended prematurely" after 413s) and not the consequence — Section 2 built
    from four models instead of five, with consensus counts read against a
    threshold the run no longer met the same way.
    """

    def _report(self, **overrides):
        return _base_report(
            model_failures=["openai:fact_check"],
            model_failure_details=[
                {
                    "pass": "openai:fact_check",
                    "model": "gpt-5.5",
                    "domain": "fact_check",
                    "section": "SECTION 2: Factual Verification",
                    "error": "Response ended prematurely",
                    "elapsed_seconds": 413.23,
                }
            ],
            **overrides,
        )

    def test_the_reason_is_reported(self):
        md = render_report_markdown(self._report())
        assert "Response ended prematurely" in md

    def test_the_model_and_elapsed_time_are_reported(self):
        md = render_report_markdown(self._report())
        assert "gpt-5.5" in md
        assert "413s" in md

    def test_the_affected_section_is_named(self):
        md = render_report_markdown(self._report())
        assert "SECTION 2: Factual Verification was built without this model" in md

    def test_the_affected_section_carries_its_own_note(self):
        """Reading Section 2 should not require having read the header."""
        md = render_report_markdown(self._report())
        section = md.split("## SECTION 2")[1].split("## SECTION 3")[0]
        assert "Built without gpt-5.5" in section

    def test_an_unaffected_section_is_not_annotated(self):
        md = render_report_markdown(self._report())
        section = md.split("## SECTION 3")[1].split("## SECTION 4")[0]
        assert "Built without" not in section

    def test_each_domain_gets_its_own_note(self):
        report = _base_report(
            model_failures=["grok:red_team"],
            model_failure_details=[
                {
                    "pass": "grok:red_team",
                    "model": "grok-4",
                    "domain": "red_team",
                    "section": "SECTION 6: Red Team Findings",
                    "error": "timeout",
                    "elapsed_seconds": 900,
                }
            ],
        )
        md = render_report_markdown(report)
        section = md.split("## SECTION 6")[1]
        assert "Built without grok-4" in section

    def test_every_named_section_exists_in_the_rendered_report(self):
        """Drift guard.

        The section names live in consolidation but are the renderer's own
        headings. Telling a reader "SECTION 6: Red Team was built without this
        model" when the heading says "SECTION 6: Red Team Findings" sends them
        looking for something that is not there.
        """
        md = render_report_markdown(_base_report())
        for domain, section in _DOMAIN_SECTIONS.items():
            assert f"## {section}" in md, (
                f"_DOMAIN_SECTIONS[{domain!r}] names {section!r}, which the "
                f"report does not render as a heading."
            )

    def test_a_clean_run_says_nothing(self):
        md = render_report_markdown(_base_report())
        assert "Failed model passes" not in md
        assert "Built without" not in md

    def test_a_report_predating_the_details_still_renders(self):
        """Old reports on disk are re-rendered by history; they have the
        bare list and no details."""
        md = render_report_markdown(_base_report(model_failures=["openai:fact_check"]))
        assert "openai:fact_check" in md
        assert "Failed model passes" in md


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


class TestSection2EvidenceCoverage:
    """A `confirmed` verdict is a model's judgment. Until the prompt asked for a
    verbatim quote and a direct URL, nothing in the output told a reader whether
    it rested on anything openable — 85 confirmed claims in the 2026-08-12 run,
    50 with no URL at all.
    """

    def _fc(self, *items):
        return {"confirmed": list(items)}

    def test_full_evidence_is_reported_without_a_caveat(self):
        md = render_report_markdown(
            _base_report(
                section_2_fact_check=self._fc(
                    {
                        "claim": "a",
                        "supporting_quote": "the page says a",
                        "source_url": "https://example.gov/a",
                    }
                )
            )
        )
        assert "1 of 1 verdict(s) arrived with a verbatim supporting quote" in md
        assert "1 with a direct source URL" in md
        assert "the model's assertion rather than" not in md

    def test_missing_evidence_is_counted_and_explained(self):
        md = render_report_markdown(
            _base_report(
                section_2_fact_check=self._fc(
                    {
                        "claim": "a",
                        "supporting_quote": "q",
                        "source_url": "https://example.gov/a",
                    },
                    {"claim": "b", "source": "Some Publisher"},
                    {"claim": "c"},
                )
            )
        )
        assert "1 of 3 verdict(s) arrived with a verbatim supporting quote" in md
        assert "1 with a direct source URL" in md
        assert "the model's assertion rather than" in md

    def test_whitespace_only_evidence_does_not_count(self):
        md = render_report_markdown(
            _base_report(
                section_2_fact_check=self._fc(
                    {"claim": "a", "supporting_quote": "   ", "source_url": "  "}
                )
            )
        )
        assert "0 of 1 verdict(s) arrived with a verbatim supporting quote" in md

    def test_declining_to_reach_a_verdict_is_not_counted_as_missing_evidence(self):
        """unverifiable and primary_source_needed are the honest answer when the
        model has nothing to quote. Counting them here would penalise exactly
        the behaviour the prompt asks for."""
        report = _base_report(
            section_2_fact_check={
                "unverifiable": [{"claim": "a", "reason": "paywalled"}],
                "primary_source_needed": [{"claim": "b", "best_candidate_source": "x"}],
            }
        )
        md = render_report_markdown(report)
        assert "verdict(s) arrived with" not in md

    def test_the_quote_reaches_the_reader(self):
        md = render_report_markdown(
            _base_report(
                section_2_fact_check=self._fc(
                    {"claim": "a", "supporting_quote": "Illinois generated 53%."}
                )
            )
        )
        assert "Illinois generated 53%." in md


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
        assert "### Read, and supports the claim" in md
        assert "### No source identified" in md
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
        assert "### Read, and supports the claim" in md
        assert "### Pointer only" in md
        assert "not independently verified" in md.lower()

        verified_section = md.split("### Read, and supports the claim")[1].split(
            "### Pointer only"
        )[0]
        pointer_section = md.split("### Pointer only")[1]
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

        assert "### Fetched, but could not be read" in md
        assert "could not be read" in md.lower()

        unverifiable_section = md.split("### Fetched, but could not be read")[1].split(
            "###"
        )[0]
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

        assert "| Fetched, but could not be read | 1 |" in md

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

        drift_section = md.split("### ⚠ Content changed")[1].split(
            "### Read, and supports the claim"
        )[0]
        assert "https://example.gov/budget.pdf" in drift_section
        assert "run 4 of 'prior-article' on 2026-01-01T00:00:00" in drift_section
        assert "may need re-checking" in drift_section
        # The entry still renders in its tier; it is not moved out of the
        # confirmed block.
        assert (
            "The bridge cost $4 million."
            in md.split("### Read, and supports the claim")[1]
        )

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

    def test_lead_states_the_checked_fraction_before_any_tier(self):
        """The reader's first number must be "how many were actually checked",
        not a four-way tier breakdown they have to do arithmetic on. In the run
        that motivated this, 9 of 144 claims had a document fetched and read and
        the section opened with "9 verified, 15 pointer-only, ..." — accurate,
        but it buried the 6%."""
        citations = [
            {"claim": f"checked {i}", "resolved": True, "verification": "checksum"}
            for i in range(3)
        ] + [{"claim": f"nothing {i}", "resolved": False} for i in range(7)]
        md = render_report_markdown(_base_report(section_9_citations=citations))

        lead = md.split("## SECTION 9")[1].split("|")[0]
        assert "3 of 10 claim(s) (30%) were checked" in lead
        assert "research lead, not a verification" in lead

    def test_every_claim_lands_in_exactly_one_disposition_row(self):
        """The table is the section's accounting. If the rows don't sum to the
        total, some claims are being counted twice or not at all — which is the
        precise failure the old "unresolved" bucket had, since it overlapped the
        content-mismatch entries."""
        citations = [
            {"claim": "a", "resolved": True, "verification": "checksum"},
            {"claim": "b", "resolved": False, "verification": "content_mismatch"},
            {"claim": "c", "resolved": True, "verification": "unverifiable"},
            {"claim": "d", "resolved": True, "verification": "pointer"},
            {"claim": "e", "resolved": False},
            # resolved:True but no tier — still "no source retrieved", because
            # nothing was fetched to read.
            {"claim": "f", "resolved": True},
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))

        counts = [int(n) for n in re.findall(r"^\| .+ \| (\d+) \|$", md, re.M)]
        assert sum(counts) == len(citations)
        # checksum, mismatch, unverifiable, fetch_failed, pointer, no_source
        assert counts == [1, 1, 1, 0, 1, 2]

    def test_content_mismatch_is_not_buried_with_never_looked_up_claims(self):
        """A source fetched, read, and found not to support the claim is the
        highest-information row in the section. It used to render inside
        "Unresolved" alongside claims nothing was ever retrieved for, which made
        the most actionable finding a run produces indistinguishable from a gap.
        """
        citations = [
            {
                "claim": "IARC classified ELF-EMF as Group 2B in 2002.",
                "resolved": False,
                "url": "https://publications.iarc.who.int/100",
                "verification": "content_mismatch",
                "relevance_verdict": "not_addressed",
                "note": "does not mention ELF-EMF",
            },
            {"claim": "The plan was approved in 2021.", "resolved": False},
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))

        assert "### Read, and does NOT support the claim (1)" in md
        mismatch_section = md.split("### Read, and does NOT support the claim")[
            1
        ].split("###")[0]
        assert "IARC" in mismatch_section
        # The claim nothing was fetched for must not be in there with it.
        assert "The plan was approved in 2021." not in mismatch_section
        assert (
            "The plan was approved in 2021." in md.split("### No source identified")[1]
        )

    def test_mismatch_distinguishes_contradicts_from_not_addressed(self):
        """ "The source refutes you" and "the source doesn't cover this" warrant
        different reactions — the first is a possible factual error, the second
        is usually the wrong URL. Collapsing them overstates one and buries the
        other."""
        not_addressed = [
            {
                "claim": "a",
                "resolved": False,
                "verification": "content_mismatch",
                "relevance_verdict": "not_addressed",
            }
        ]
        md = render_report_markdown(_base_report(section_9_citations=not_addressed))
        assert "None came back `contradicts`" in md
        assert "wrong URL was checked" in md

        contradicts = not_addressed + [
            {
                "claim": "b",
                "resolved": False,
                "verification": "content_mismatch",
                "relevance_verdict": "contradicts",
            }
        ]
        md = render_report_markdown(_base_report(section_9_citations=contradicts))
        assert "1 of these came back `contradicts`" in md
        assert "possible factual error" in md

    def test_a_conceding_reask_changes_the_guidance(self):
        """Whenever nothing came back `contradicts`, the block told the reader
        the page simply had not covered the claim and the URL was probably
        wrong. On the Honda run the one `not_addressed` entry carried a re-ask
        in which the asserting model read its own refutation, concluded the
        *claim* was wrong, and proposed the fix — and the guidance steered the
        reader away from the only actionable finding in the block."""
        plain = [_mismatch_citation()]
        md = render_report_markdown(_base_report(section_9_citations=plain))
        assert "None came back `contradicts`" in md
        assert "wrong URL was checked" in md

        conceded = [
            _mismatch_citation(
                reask={
                    "action": "correct_claim",
                    "asked_model": "gemini",
                    "reason": "The page says 1998 or 2002.",
                    "corrected_claim": "the chip advanced to January 1, 1998 or 2002.",
                }
            )
        ]
        md = render_report_markdown(_base_report(section_9_citations=conceded))
        assert "concluded the claim — not the citation — was wrong" in md
        # Replaced, not merely appended to: the sentence that contradicted the
        # finding below it must not survive alongside the correction.
        assert "None came back `contradicts`" not in md
        assert "wrong URL was checked" not in md

    def test_a_withdrawing_reask_concedes_the_claim_too(self):
        """`withdraw` faults the claim as squarely as `correct_claim` does; it
        just has no replacement wording to offer."""
        md = render_report_markdown(
            _base_report(
                section_9_citations=[
                    _mismatch_citation(
                        reask={
                            "action": "withdraw",
                            "asked_model": "gemini",
                            "reason": "Nothing supports it.",
                        }
                    )
                ]
            )
        )
        assert "concluded the claim — not the citation — was wrong" in md
        assert "wrong URL was checked" not in md

    def test_a_reask_that_blames_the_source_leaves_the_guidance_alone(self):
        """`different_source` stands by the claim and blames the URL — which is
        precisely the citation problem the default guidance describes — and
        `stand` concedes nothing at all. Neither may flip the header."""
        for action in ("different_source", "stand"):
            md = render_report_markdown(
                _base_report(
                    section_9_citations=[
                        _mismatch_citation(
                            reask={
                                "action": action,
                                "asked_model": "gemini",
                                "reason": "r",
                                "source_url": "https://example.test/proposed",
                            }
                        )
                    ]
                )
            )
            assert "None came back `contradicts`" in md, action
            assert "not the citation — was wrong" not in md, action

    def test_each_guidance_sentence_is_scoped_to_the_entries_it_covers(self):
        """With a mix in the block, every sentence has to carry its own count —
        the reader is deciding what to do per entry, not per section."""
        citations = [
            _mismatch_citation(claim="contradicted", relevance_verdict="contradicts"),
            _mismatch_citation(
                claim="conceded",
                reask={
                    "action": "correct_claim",
                    "asked_model": "gemini",
                    "corrected_claim": "fixed wording",
                },
            ),
            _mismatch_citation(claim="plain one"),
            _mismatch_citation(claim="plain two"),
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))
        assert "1 of these came back `contradicts`" in md
        assert "refutation for 1 of these" in md
        assert "The remaining 2 entries are" in md
        # The unconditional wording must never appear beside a scoped count.
        assert "None came back `contradicts`" not in md

    def test_the_proposed_correction_precedes_the_bulk_evidence(self):
        """The correction was the last line of a 2,682-character entry, below a
        546-character content summary and three opaque redirect URLs."""
        md = render_report_markdown(
            _base_report(
                section_9_citations=[
                    _mismatch_citation(
                        reask={
                            "action": "correct_claim",
                            "asked_model": "gemini",
                            "corrected_claim": "PROPOSED WORDING HERE",
                        }
                    )
                ]
            )
        )
        entry = md.split("### Read, and does NOT support the claim")[1]
        correction = entry.index("PROPOSED WORDING HERE")
        for later in ("Content summary:", "Checksum:", "Alternates checked:"):
            assert correction < entry.index(later), later

    def test_reordering_keeps_every_piece_of_evidence_reachable(self):
        """The section exists so a reader can audit a tier instead of taking it
        on trust. Reordering and compacting are fine; dropping evidence is not."""
        md = render_report_markdown(
            _base_report(section_9_citations=[_mismatch_citation()])
        )
        assert "451bb861017fe7c4446b7dc274a2d6ce" in md
        assert "The page discusses 1998 or 2002, not 2003." in md
        assert "https://web.archive.test/x" in md
        assert "Live: https://example.test/redirect/" in md
        for alternate in ("alt-one", "alt-two"):
            assert alternate in md, alternate

    def test_alternates_render_as_a_count_with_the_urls_beneath(self):
        """Three opaque redirect URLs run together inside a Python list repr
        were 546 of one entry's 2,682 characters, and the count they add up to
        was not readable without parsing them."""
        md = render_report_markdown(
            _base_report(section_9_citations=[_mismatch_citation()])
        )
        assert "Alternates checked: 2 other source(s)" in md
        assert "['https://example.test/alt-one/" not in md

    def test_alternates_under_a_supported_claim_do_not_impugn_the_primary(self):
        """`alternates_checked` is set on checksum entries too — five of them in
        the saved runs — and there the primary source *did* support the claim.
        The resolver's note says "none supported it either", which is true only
        where it is written. This renderer is shared, so it may say only what
        holds of the alternates themselves."""
        supported = _mismatch_citation(
            verification="checksum",
            relevance_verdict="supports",
            relevance_reason="The page states it directly.",
            note="",
        )
        md = render_report_markdown(_base_report(section_9_citations=[supported]))
        assert "### Read, and supports the claim (1)" in md
        assert "Alternates checked: 2 other source(s)" in md
        assert "none of them supported it." in md
        assert "either" not in md.split("Alternates checked:")[1].split("\n")[0]

    def test_note_is_dropped_only_when_it_restates_the_reason_above_it(self):
        """The resolver builds the note as verdict + reason + alternates count,
        and all three now render as their own lines. A note that says anything
        else is still evidence, so the test is on the string, not the tier."""
        md = render_report_markdown(
            _base_report(section_9_citations=[_mismatch_citation()])
        )
        assert "Relevance reason: The page discusses 1998 or 2002, not 2003." in md
        assert "Note: Source URL loaded" not in md

        md = render_report_markdown(
            _base_report(
                section_9_citations=[
                    _mismatch_citation(note="Publisher issued a correction in 2024.")
                ]
            )
        )
        assert "Note: Publisher issued a correction in 2024." in md

    def test_confirmed_bucket_is_reconciled_against_what_was_retrieved(self):
        """The fact-check pass's "confirmed" verdict and this section's
        retrieval are independent, and a reader seeing "Bucket: confirmed" beside
        a claim with no URL reads it as corroboration. In the motivating run 85
        claims came back confirmed and 50 of those had nothing fetched at all."""
        citations = [
            {
                "claim": "a",
                "resolved": True,
                "verification": "checksum",
                "fact_check_bucket": "confirmed",
            },
        ] + [
            {"claim": f"b{i}", "resolved": False, "fact_check_bucket": "confirmed"}
            for i in range(4)
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))

        assert 'called 5 of these claims "confirmed."' in md
        assert "1 had a document fetched and read here that supports the claim" in md
        assert "for 4 no document was read at all" in md

    def test_checked_count_includes_mismatches_not_just_confirmations(self):
        """ "Checked" means a document was fetched and read — which is true of a
        mismatch too; the check simply returned "no". Counting only the
        confirmations as checked would repeat, in the summary line, the exact
        conflation between "a model says so" and "a document was read" that the
        tiers below exist to separate."""
        citations = [
            {"claim": "a", "resolved": True, "verification": "checksum"},
            {"claim": "b", "resolved": False, "verification": "content_mismatch"},
            {"claim": "c", "resolved": False},
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))

        assert "2 of 3 claim(s) (67%) were checked" in md
        assert "1 where the document supported the claim, 1 where it did not" in md

    def test_refused_fetch_is_not_reported_as_no_source(self):
        """A URL that was identified and refused (403/404/DNS) is a different
        fact about a claim than never having had a URL, and a more actionable
        one: a publisher that blocks an automated fetch usually still serves the
        page to a person. Both used to land in one "Unresolved" pile."""
        citations = [
            {
                "claim": "The ASME study measured neighbourhood warming.",
                "resolved": False,
                "url": "https://asmedigitalcollection.asme.org/article/7/2/024501",
                "note": "Known source URL could not be fetched: 403 Client Error",
            },
            {"claim": "Nothing was ever found for this one.", "resolved": False},
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))

        assert "### Source URL identified, but the fetch was refused (1)" in md
        refused = md.split("### Source URL identified, but the fetch was refused")[
            1
        ].split("###")[0]
        assert "asmedigitalcollection" in refused
        assert "Nothing was ever found" not in refused

        assert "### No source identified (1)" in md
        assert "Nothing was ever found" in md.split("### No source identified")[1]

    def test_refused_fetch_is_paired_with_its_archive_copy(self):
        """The archive pairing was added for the tiers that existed at the time.
        A refused fetch is the case that needs it most — the live URL is exactly
        the one that did not load, so the snapshot may be the only readable copy
        — and that bucket did not exist to be wired up."""
        citations = [
            {
                "claim": "The ASME study measured neighbourhood warming.",
                "resolved": False,
                "url": "https://asmedigitalcollection.asme.org/article/7/2/024501",
                "note": "Known source URL could not be fetched: 403 Client Error",
                "wayback": {
                    "archived": True,
                    "snapshot_url": "https://web.archive.org/web/2026/asme",
                },
            }
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))
        refused = md.split("### Source URL identified, but the fetch was refused")[
            1
        ].split("###")[0]
        assert "https://web.archive.org/web/2026/asme" in refused
        assert "Cite both" in refused

    def test_detail_blocks_render_in_the_same_order_as_the_table(self):
        """The table is the section's index. If the blocks below it run in a
        different order the reader cannot navigate from one to the other, which
        is most of what the table is for."""
        citations = [
            {"claim": "a", "resolved": True, "verification": "checksum"},
            {"claim": "b", "resolved": False, "verification": "content_mismatch"},
            {"claim": "c", "resolved": True, "verification": "unverifiable"},
            {"claim": "d", "resolved": False, "url": "https://example.gov/403"},
            {"claim": "e", "resolved": True, "verification": "pointer"},
            {"claim": "f", "resolved": False},
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))
        section = md.split("## SECTION 9")[1]

        # A table label may carry a trailing gloss its heading does not repeat
        # ("Pointer only — nothing retrieved"), so match on the stable prefix.
        heads = [
            label.split(" — ")[0]
            for _, label in report_markdown._DISPOSITIONS  # noqa: SLF001
        ]
        positions = [section.index(f"### {h}") for h in heads]
        assert positions == sorted(positions), dict(zip(heads, positions))

    def test_wholesale_archive_lookup_failure_is_stated_once(self):
        """Per-entry "NOT CHECKED" is invisible in aggregate: a reader skimming
        a page of them concludes nothing is archived. The circuit breaker makes
        this the common case — once archive.org 429s enough, the run stops
        asking and every remaining citation carries a null."""
        citations = [
            {
                "claim": f"c{i}",
                "resolved": True,
                "verification": "checksum",
                "url": f"https://example.gov/{i}",
                "wayback": {"archived": None, "rate_limited": True},
            }
            for i in range(3)
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))

        assert "Archive status is unknown for 3 of these citations" in md
        assert "rate-limited this run (HTTP 429)" in md
        assert "**not** that the page is unarchived" in md

    def test_partial_archive_failure_reports_the_rate_limited_share(self):
        citations = [
            {
                "claim": "a",
                "resolved": True,
                "verification": "checksum",
                "url": "https://example.gov/a",
                "wayback": {"archived": None, "rate_limited": True},
            },
            {
                "claim": "b",
                "resolved": True,
                "verification": "checksum",
                "url": "https://example.gov/b",
                "wayback": {"archived": None},
            },
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))
        assert "unknown for 2 of these citations" in md
        assert "1 of them to archive.org rate limiting" in md

    def test_a_known_unarchived_page_is_not_called_unknown(self):
        """archived:False is an answer. Only null is "we did not find out"."""
        citations = [
            {
                "claim": "a",
                "resolved": True,
                "verification": "checksum",
                "url": "https://example.gov/a",
                "wayback": {"archived": False},
            }
        ]
        md = render_report_markdown(_base_report(section_9_citations=citations))
        assert "Archive status is unknown" not in md

    def test_no_confirmed_reconciliation_when_the_bucket_is_absent(self):
        """Reports whose fact-check pass produced no confirmed bucket must not
        grow an empty callout."""
        citations = [{"claim": "a", "resolved": True, "verification": "checksum"}]
        md = render_report_markdown(_base_report(section_9_citations=citations))
        assert "The fact-check pass called" not in md
        # The disposition label still says "confirmed" — that is the tier name,
        # not the reconciliation callout.
        assert "| Read, and supports the claim | 1 |" in md


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


def _currency(live=None, **overrides):
    block = {
        "warnings": [],
        "notices": [],
        "registry_date": "2026-06-22",
        "registry_age_days": 55,
        "registry_stale": False,
        "registry_warning": False,
    }
    block.update(overrides)
    if live is not None:
        block["live"] = live
    return block


class TestModelCurrencySection:
    """How old the models behind this report are.

    The report carried `model_currency` in its JSON from the start and rendered
    none of it — the readable review, which is the artifact a human actually
    opens and pastes into a revision session, said nothing about model age at
    all. And the registry it came from can only name a replacement somebody
    already wrote into model_registry.yaml, so "you ran gpt-5.5 and gpt-5.6
    shipped last week" was unsayable by construction.
    """

    def test_a_report_without_the_block_renders_exactly_as_before(self):
        """Reports written before this section must be unaffected."""
        md = render_report_markdown(_base_report())
        assert "Model Currency" not in md

    def test_a_newer_model_is_named_against_the_model_that_ran(self):
        md = render_report_markdown(
            _base_report(
                model_currency=_currency(
                    live={
                        "newer": [
                            {
                                "provider": "openai",
                                "model": "gpt-5.5",
                                "newer": [
                                    {
                                        "model": "gpt-5.6",
                                        "released": "2026-08-10",
                                        "price_known": True,
                                    }
                                ],
                                "undated_models": 0,
                            }
                        ],
                        "current": [],
                        "unchecked": [],
                    }
                )
            )
        )
        assert "## Model Currency" in md
        assert "gpt-5.5" in md and "gpt-5.6" in md
        assert "2026-08-10" in md

    def test_an_unpriced_new_model_says_so_rather_than_quoting_a_guess(self):
        """pricing.yaml is hand-maintained; a week-old model is what it misses.

        Reporting the unknown-model fallback as though it were the rate would
        present a placeholder as a price.
        """
        md = render_report_markdown(
            _base_report(
                model_currency=_currency(
                    live={
                        "newer": [
                            {
                                "provider": "openai",
                                "model": "gpt-5.5",
                                "newer": [
                                    {
                                        "model": "gpt-5.6",
                                        "released": "2026-08-10",
                                        "price_known": False,
                                    }
                                ],
                                "undated_models": 0,
                            }
                        ],
                        "current": [],
                        "unchecked": [],
                    }
                )
            )
        )
        assert "pricing.yaml" in md
        assert "fallback" in md

    def test_the_section_does_not_recommend_switching(self):
        """House style: advisory only. See analysis/seo_suggest.py."""
        md = render_report_markdown(
            _base_report(
                model_currency=_currency(
                    live={
                        "newer": [
                            {
                                "provider": "openai",
                                "model": "gpt-5.5",
                                "newer": [
                                    {
                                        "model": "gpt-5.6",
                                        "released": "2026-08-10",
                                        "price_known": True,
                                    }
                                ],
                                "undated_models": 0,
                            }
                        ],
                        "current": [],
                        "unchecked": [],
                    }
                )
            )
        )
        normalized = " ".join(md.split())
        assert "not automatically a better or a cheaper one" in normalized
        assert "not what you should switch to" in normalized

    def test_an_unchecked_provider_is_not_reported_as_up_to_date(self):
        """The conflation the section exists to prevent."""
        md = render_report_markdown(
            _base_report(
                model_currency=_currency(
                    live={
                        "newer": [],
                        "current": [{"provider": "grok", "model": "grok-4.3"}],
                        "unchecked": [
                            {
                                "provider": "openai",
                                "model": "gpt-5.5",
                                "reason": "the models API returned HTTP 401",
                            }
                        ],
                    }
                )
            )
        )
        assert "Not checked" in md
        assert "HTTP 401" in md
        # grok was checked and is current; openai must not be swept into that.
        current_line = next(
            line for line in md.splitlines() if "nothing newer offered" in line
        )
        assert "grok" in current_line and "openai" not in current_line

    def test_no_discovery_data_at_all_is_one_line_not_a_roll_call(self):
        """The default path prints on every run — it has to stay short."""
        md = render_report_markdown(
            _base_report(
                model_currency=_currency(
                    live={
                        "newer": [],
                        "current": [],
                        "unchecked": [
                            {
                                "provider": p,
                                "model": f"{p}-model",
                                "reason": "never checked",
                            }
                            for p in ("openai", "gemini", "mistral", "grok", "claude")
                        ],
                    }
                )
            )
        )
        assert "ci-discover" in md
        # Not one bullet per provider.
        assert md.count("never checked") == 0

    def test_registry_findings_render_without_any_live_data(self):
        """The registry half stands alone — live discovery is opt-in."""
        md = render_report_markdown(
            _base_report(
                model_currency=_currency(
                    warnings=[
                        {
                            "provider": "openai",
                            "model": "gpt-4o",
                            "replacement": "gpt-5.4",
                            "note": "GPT-5 family available",
                        }
                    ]
                )
            )
        )
        assert "## Model Currency" in md
        assert "gpt-4o" in md and "gpt-5.4" in md

    def test_a_stale_registry_says_so(self):
        md = render_report_markdown(
            _base_report(
                model_currency=_currency(
                    registry_age_days=200,
                    registry_stale=True,
                    registry_warning=True,
                    notices=[
                        {
                            "provider": "claude",
                            "model": "claude-sonnet-4-6",
                            "newer": "claude-opus-4-8",
                            "note": "",
                        }
                    ],
                )
            )
        )
        assert "overdue for review" in md
        assert "200 days ago" in md


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

    def test_a_lookup_that_never_completed_is_not_reported_as_unarchived(self):
        """``archived: None`` is "we never found out", not "there is no snapshot".

        Reporting it as "none" asserts something the run never established. The
        rate-limit circuit breaker makes this the common case rather than a rare
        one — once it trips, every remaining citation carries a null.
        """
        out = self._text(self._cit(archived=None, error="skipped: rate limit"))
        assert "NOT CHECKED" in out
        assert "only as durable as the live URL" not in out
        assert "Archive: none" not in out

    def test_the_circuit_breaker_state_renders_as_not_checked(self):
        """End to end: what the breaker actually returns, through the renderer.

        Guards the two halves against drifting apart — the breaker's return shape
        is what decides which branch this function takes.
        """
        wayback.reset_rate_limit_state()
        wayback._rate_limited_lookups = wayback._CIRCUIT_TRIP_AFTER
        try:
            result = wayback.check("https://example.org/page")
        finally:
            wayback.reset_rate_limit_state()
        citation = {
            "claim": "A claim",
            "url": "https://example.org/page",
            "resolved": True,
            "wayback": result,
        }
        out = "\n".join(report_markdown._render_archive_pair(citation))
        assert "NOT CHECKED" in out
        assert "Archive: none" not in out

    def test_a_citation_with_no_url_renders_no_pairing(self):
        citation = {"claim": "c", "resolved": True, "verification": "checksum"}
        assert report_markdown._render_archive_pair(citation) == []


class TestUnresolvedCitationsRenderTheirArchiveState:
    """A failed fetch now carries archive.org's answer, or deliberately no key.

    The resolver used to drop the availability result on the failure path, so an
    unresolved citation recorded no archive state at all. ``_render_archive_pair``
    then read ``citation.get("wayback") or {}`` — and ``{}.get("archived")`` is
    None — so every unresolved citation claimed "the archive.org lookup did not
    complete this run", including the 404s where nothing was ever looked up.

    Three states, three answers: archive.org said no, archive.org never
    answered, and archive.org was never asked. The third is the one an absent
    key means, and it is not a re-runnable failure — nothing will ask next time
    either.
    """

    LIVE = "https://example.org/page"

    def _cit(self, **overrides):
        """An unresolved citation carrying a URL — the `fetch_failed` bucket."""
        citation = {
            "claim": "A claim",
            "url": self.LIVE,
            "resolved": False,
            "note": "Known source URL could not be fetched: timed out",
        }
        citation.update(overrides)
        return citation

    def _section(self, *citations):
        return "\n".join(report_markdown._render_section_9(list(citations)))

    def test_a_known_absent_snapshot_says_archive_org_answered(self):
        out = self._section(self._cit(wayback={"archived": False}))
        assert "archive.org answered and has no snapshot" in out
        assert "NOT CHECKED" not in out
        assert "NOT LOOKED UP" not in out

    def test_an_unresolved_citation_is_not_promised_archiving_it_will_not_get(self):
        """`_submit_missing_archives` requires `resolved`, so "re-run once
        archiving succeeds" is a promise nothing in the pipeline keeps here. Nor
        is there a fetched copy for the live URL to be "as durable as"."""
        out = self._section(self._cit(wayback={"archived": False}))
        assert "re-run once archiving succeeds" not in out
        assert "only as durable as the live URL" not in out
        assert "not submitted for archiving" in out

    def test_a_resolved_citation_keeps_the_durability_wording(self):
        """The unresolved rewording must not leak into the case it is not for."""
        out = self._section(
            self._cit(
                resolved=True, verification="checksum", wayback={"archived": False}
            )
        )
        assert "only as durable as the live URL" in out
        assert "archive.org answered and has no snapshot" not in out

    def test_a_lookup_that_never_completed_still_says_not_checked(self):
        out = self._section(
            self._cit(wayback={"archived": None, "error": "rate limit tripped"})
        )
        assert "NOT CHECKED" in out
        assert "archive.org answered" not in out
        # And it reaches the aggregate callout, which is the point of carrying
        # `wb` out at all: a throttled run's unresolved citations were invisible
        # to this count before.
        assert "Archive status is unknown for 1 of these citations" in out

    def test_a_fetch_nobody_looked_up_is_not_reported_as_not_checked(self):
        """No `wayback` key: a 404/5xx, or an address the SSRF guard refused.

        "NOT CHECKED" would imply a lookup that could succeed on a re-run. None
        was attempted, and none will be.
        """
        out = self._section(self._cit())
        assert "NOT LOOKED UP" in out
        assert "NOT CHECKED" not in out
        assert "Archive status is unknown" not in out

    def test_an_unreadable_snapshot_is_still_offered_to_the_reader(self):
        """archive.org has a copy; the pipeline just could not fetch it. That
        copy may be the only readable version of a refused URL, so it is the
        most useful thing the run learned — and the old contract dropped it."""
        snapshot = "https://web.archive.org/web/20240101000000/https://example.org/page"
        out = self._section(
            self._cit(
                wayback={
                    "archived": True,
                    "snapshot_url": snapshot,
                    "snapshot_age_days": 245,
                    "snapshot_stale": True,
                }
            )
        )
        assert f"Archive: {snapshot}" in out
        assert "STALE" in out
        assert f"Cite both: {self.LIVE} (archived: {snapshot})" in out

    def test_the_wayback_line_reads_for_a_human_not_a_debugger(self):
        """The new key flows through `_kv_lines` too, which must not dump it raw."""
        out = self._section(self._cit(wayback={"archived": False}))
        assert "Wayback: Not archived in Wayback Machine" in out
        assert "{'archived'" not in out

    def test_never_asked_is_not_counted_in_the_unknown_aggregate(self):
        """End to end through the real renderer, mixing the two failure routes:
        only the citation that was actually looked up is unknown."""
        md = render_report_markdown(
            _base_report(
                section_9_citations=[
                    self._cit(claim="looked up", wayback={"archived": None}),
                    self._cit(claim="never asked"),
                ]
            )
        )
        assert "Archive status is unknown for 1 of these citations" in md
        assert "NOT LOOKED UP" in md
        assert "NOT CHECKED" in md


#: Standing in for the sentence ``resolver._resolve_known_url`` writes onto an
#: escalated citation. The renderer passes it through rather than composing its
#: own, so one route states itself once — the wording itself is asserted in
#: ``test_resolver.py``, where it is written.
_READER_ACCESS = (
    "This source refused an ordinary automated request (403) and served the "
    "page only to a browser-shaped client. …cite the archive copy alongside it."
)


class TestEscalatedCitationsRenderReaderFriction:
    """``verified_via`` is reader-facing information, not a disclosure.

    For a public document the retrieval method does not bear on whether the
    citation is valid — the reader opening the link gets the same page. What it
    does bear on is durability: a source that refused an ordinary automated
    request is the one whose link is most likely to fail somebody later, which
    makes it exactly the citation that needs an archive copy beside it. That is
    the framing under test here, in both directions — it has to say enough to be
    useful and not so much that it reads as an admission.
    """

    LIVE = "https://www.bianchihonda.com/change-clock-with-navigation-repair/"
    SNAPSHOT = "http://web.archive.org/web/20250912181258/" + LIVE

    def _cit(self, **overrides):
        citation = {
            "claim": "The placeholder date was January 1, 2002.",
            "url": self.LIVE,
            "resolved": True,
            "verification": "checksum",
            "verified_via": "tls_impersonation",
            "origin_failure": "blocked",
            "reader_access": _READER_ACCESS,
            "wayback": {"archived": True, "snapshot_url": self.SNAPSHOT},
        }
        citation.update(overrides)
        return citation

    def _pair(self, **overrides):
        return "\n".join(report_markdown._render_archive_pair(self._cit(**overrides)))

    def test_the_friction_is_stated_where_the_links_are(self):
        out = self._pair()
        assert f"Reader access: {_READER_ACCESS}" in out
        # Immediately under the live URL, not paragraphs away in a key dump:
        # this is what the author needs while deciding what to paste.
        lines = [line.strip() for line in out.splitlines()]
        assert lines[0].startswith("- Live:")
        assert lines[1].startswith("- Reader access:")

    def test_an_escalated_citation_still_offers_the_paste_ready_pairing(self):
        out = self._pair()
        assert f"Cite both: {self.LIVE} (archived: {self.SNAPSHOT})" in out

    def test_a_missing_archive_is_stated_and_not_softened(self):
        """The absence is the finding.

        An escalated citation with no snapshot is the weakest thing this section
        can produce: the live URL already refused a client once, so "only as
        durable as the live URL" — the wording every other unarchived citation
        gets — understates it.
        """
        out = self._pair(wayback={"archived": False})
        assert "Archive: NONE" in out
        assert "most needed one" in out
        assert "only as durable as" not in out

    def test_an_ordinary_unarchived_citation_keeps_its_milder_wording(self):
        out = self._pair(
            verified_via="direct", reader_access=None, wayback={"archived": False}
        )
        assert "Archive: none." in out
        assert "Archive: NONE" not in out
        assert "Reader access:" not in out

    def test_a_directly_fetched_citation_says_nothing_about_retrieval(self):
        """The unremarkable case stays silent. A line on every entry reporting
        that nothing happened would bury the two entries where something did."""
        out = self._pair(verified_via="direct", reader_access=None)
        assert "Reader access:" not in out

    def test_an_archive_read_states_its_provenance_exactly_once(self):
        """Each retrieval route has one sentence, and only one.

        A citation read from the archive already carries ``archive_provenance``,
        which says more than a generic line could — it names the origin failure
        and flags a stale snapshot. Adding a second sentence for the same fact
        would make the entry state its provenance twice.
        """
        citation = self._cit(
            verified_via="wayback_fallback",
            reader_access=None,
            archive_provenance="Content was read from an archive.org snapshot…",
        )
        pair = "\n".join(report_markdown._render_archive_pair(citation))
        assert "Reader access:" not in pair

        md = "\n".join(report_markdown._render_section_9([citation]))
        assert md.count("Content was read from an archive.org snapshot") == 1
        assert "wayback_fallback" not in md

    def test_an_escalated_fetch_feeds_the_redirector_fix(self):
        """The escalation path has to record ``final_url`` like the direct one.

        Every refused claim in the 2026-09-05 Honda run cited a Vertex
        ``grounding-api-redirect`` URL, so these are exactly the citations
        ``_citation_pair`` was taught to resolve — and they only reach it
        because ``_impersonation_fallback_content`` reports where the fetch
        landed. Left unset, an escalated citation would still be published as
        271 opaque characters that expire.
        """
        redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ"
        out = self._pair(url=redirect, final_url=self.LIVE)

        assert f"Live: {self.LIVE}" in out
        assert f"Cite both: {self.LIVE} (archived: {self.SNAPSHOT})" in out
        assert redirect not in out

    def test_the_raw_enum_is_never_dumped_beside_the_prose(self):
        """``verified_via`` used to reach the reader through the generic
        key/value dump as ``Verified via: tls_impersonation``. Rendering it
        twice — once as an enum, once as prose — is worse than either."""
        md = "\n".join(report_markdown._render_section_9([self._cit()]))
        assert "tls_impersonation" not in md
        assert "Verified via:" not in md
        assert "Reader access:" in md

    def test_an_escalated_citation_sits_in_the_read_and_supports_bucket(self):
        """Escalation changes how the document was obtained, not what was
        established about it. A page read this way was read."""
        md = "\n".join(report_markdown._render_section_9([self._cit()]))
        assert "### Read, and supports the claim (1)" in md
        assert "| Source URL identified, but the fetch was refused | 0 |" in md


class TestWaybackIsRenderedForAReaderNotADebugger:
    """`_kv_lines` dumped the raw wayback dict into the report.

    A reader looking for "is this archived" got `{'archived': None, 'error':
    '...'}` and had to decode it. The null case is the one that has to be right:
    it means the lookup never completed, not that there is no snapshot.
    """

    def _cit(self, **wb):
        return {"claim": "A claim", "url": "https://example.org/p", "wayback": wb}

    def test_a_never_completed_lookup_is_not_reported_as_unarchived(self):
        out = "\n".join(
            report_markdown._kv_lines(
                self._cit(archived=None, error="rate limit tripped")
            )
        )
        assert "NOT CHECKED" in out
        assert "Not archived" not in out
        assert "{'archived'" not in out, "the raw dict must not reach the report"

    def test_a_genuinely_unarchived_page_still_says_so(self):
        out = "\n".join(report_markdown._kv_lines(self._cit(archived=False)))
        assert "Not archived in Wayback Machine" in out

    def test_an_archived_page_shows_its_snapshot(self):
        out = "\n".join(
            report_markdown._kv_lines(
                self._cit(
                    archived=True,
                    snapshot_age_days=12,
                    snapshot_url="https://web.archive.org/web/1/x",
                )
            )
        )
        assert "12d ago" in out and "https://web.archive.org/web/1/x" in out

    def test_staleness_survives_the_rendering(self):
        out = "\n".join(
            report_markdown._kv_lines(
                self._cit(archived=True, snapshot_age_days=400, snapshot_stale=True)
            )
        )
        assert "[STALE]" in out


class TestWaybackSummaryDoesNotDriftFromTheAdapter:
    """`_wayback_summary` is duplicated from `wayback.format_summary`.

    Duplicated rather than imported so this module stays a dependency-free
    renderer — importing the adapter pulls `requests` in behind it, the same
    reason `_SEO_FIELD_ORDER` is duplicated. This is the test that keeps the two
    honest, across every state including the one that matters.
    """

    CASES = [
        {"archived": None, "error": "rate limit tripped earlier this run"},
        {"archived": False},
        {
            "archived": True,
            "snapshot_age_days": 12,
            "snapshot_url": "https://web.archive.org/web/1/x",
        },
        {
            "archived": True,
            "snapshot_age_days": 400,
            "snapshot_stale": True,
            "snapshot_url": "https://web.archive.org/web/2/y",
        },
    ]

    def test_every_state_renders_identically(self):
        for wb in self.CASES:
            assert report_markdown._wayback_summary(wb) == wayback.format_summary(wb), (
                wb
            )
