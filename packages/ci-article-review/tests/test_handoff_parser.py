"""Tests for handoff_parser — raw-draft / metadata-only handoff synthesis.

Covers build_handoff_from_raw_text, parse_metadata_only, and
build_handoff_from_raw_draft_and_metadata, added alongside the --raw-draft /
--metadata CLI flags (commit 803440c) but previously untested.
"""

from ci_article_review.handoff_parser import (
    build_handoff_from_raw_draft_and_metadata,
    build_handoff_from_raw_text,
    parse_metadata_only,
)


class TestBuildHandoffFromRawText:
    def test_basic_case(self):
        text = "Just some plain article text with no headers at all."
        handoff = build_handoff_from_raw_text(text, source_name="my-draft")
        assert handoff == {
            "title": "my-draft",
            "draft": text,
            "run_number": 1,
        }

    def test_h1_title_extracted(self):
        text = "# The Real Title\n\nBody text follows here."
        handoff = build_handoff_from_raw_text(text, source_name="fallback-name")
        assert handoff["title"] == "The Real Title"
        assert handoff["draft"] == text.strip()

    def test_falls_back_to_source_name_without_h1(self):
        text = "No heading here, just body copy."
        handoff = build_handoff_from_raw_text(text, source_name="fallback-name")
        assert handoff["title"] == "fallback-name"

    def test_falls_back_to_default_source_name(self):
        text = "No heading, no source_name passed."
        handoff = build_handoff_from_raw_text(text)
        assert handoff["title"] == "Untitled"

    def test_warns_when_text_looks_like_full_handoff_banner(self, caplog):
        text = "DRAFT SUBMISSION HANDOFF\nGenerated: 2026-06-08\n\nSome content."
        with caplog.at_level("WARNING"):
            handoff = build_handoff_from_raw_text(text)
        assert any("already be a handoff document" in r.message for r in caplog.records)
        # The whole file is still treated as draft body.
        assert handoff["draft"] == text.strip()

    def test_warns_when_text_contains_primary_claim_header(self, caplog):
        text = "Some intro.\nPRIMARY CLAIM\nThis is the claim.\n"
        with caplog.at_level("WARNING"):
            build_handoff_from_raw_text(text)
        assert any("already be a handoff document" in r.message for r in caplog.records)

    def test_no_handoff_warning_for_plain_draft(self, caplog):
        text = "Just a normal article with no special headers."
        with caplog.at_level("WARNING"):
            build_handoff_from_raw_text(text)
        assert not any(
            "already be a handoff document" in r.message for r in caplog.records
        )

    def test_always_warns_that_fields_are_empty(self, caplog):
        text = "Plain draft text."
        with caplog.at_level("WARNING"):
            build_handoff_from_raw_text(text)
        assert any(
            "PRIMARY CLAIM, TARGET AUDIENCE" in r.message for r in caplog.records
        )


SAMPLE_METADATA = """DRAFT SUBMISSION HANDOFF
Generated: 2026-06-08
Pipeline run: 3
Article: How Fiber Reaches Rural Towns
Publication: mikehammett

PRIMARY CLAIM
Rural fiber buildouts are economically viable once subsidy math is right.

TARGET AUDIENCE
Municipal officials and local journalists.

PRE-DRAFT ANALYSIS SUMMARY
Steelmanned position: subsidies distort the market.

SOURCES ALREADY CITED
FCC broadband map, 2025 edition.

UNCERTAIN SECTIONS
The cost-per-mile figures in paragraph 4.

KNOWN GAPS
No discussion of satellite alternatives.

ADDITIONAL CONTEXT FOR REVIEW MODELS
Prior articles covered urban buildouts; this one is rural-focused.
"""


class TestParseMetadataOnly:
    def test_parses_all_fields(self):
        result = parse_metadata_only(SAMPLE_METADATA)
        assert result["title"] == "How Fiber Reaches Rural Towns"
        assert result["publication"] == "mikehammett"
        assert result["run_number"] == 3
        assert (
            result["primary_claim"]
            == "Rural fiber buildouts are economically viable once subsidy math is right."
        )
        assert result["target_audience"] == "Municipal officials and local journalists."
        assert (
            result["pre_draft_analysis"]
            == "Steelmanned position: subsidies distort the market."
        )
        assert result["sources_cited"] == "FCC broadband map, 2025 edition."
        assert (
            result["uncertain_sections"] == "The cost-per-mile figures in paragraph 4."
        )
        assert result["known_gaps"] == "No discussion of satellite alternatives."
        assert (
            result["additional_context"]
            == "Prior articles covered urban buildouts; this one is rural-focused."
        )
        assert "draft" not in result

    def test_missing_primary_claim_warns(self, caplog):
        text = "Article: Some Title\n\nTARGET AUDIENCE\nEveryone.\n"
        with caplog.at_level("WARNING"):
            result = parse_metadata_only(text)
        assert result["primary_claim"] == ""
        assert any("PRIMARY CLAIM" in r.message for r in caplog.records)

    def test_missing_pre_draft_analysis_debug_logs(self, caplog):
        text = (
            "Article: Some Title\n\nPRIMARY CLAIM\nThe claim.\n\n"
            "TARGET AUDIENCE\nEveryone.\n"
        )
        with caplog.at_level("DEBUG"):
            result = parse_metadata_only(text)
        assert result["pre_draft_analysis"] == ""
        assert any(
            "PRE-DRAFT ANALYSIS SUMMARY" in r.message and r.levelname == "DEBUG"
            for r in caplog.records
        )

    def test_last_section_not_lost_without_trailing_draft_marker(self):
        # Regression test for the _extract_section boundary-fix bug: a
        # metadata-only file with no trailing DRAFT marker must still
        # extract its final section (ADDITIONAL CONTEXT FOR REVIEW MODELS)
        # instead of silently losing it because the lookahead required the
        # next header to literally appear in the text.
        text = (
            "Article: Some Title\n\n"
            "PRIMARY CLAIM\nThe claim.\n\n"
            "ADDITIONAL CONTEXT FOR REVIEW MODELS\n"
            "This is the final section with no DRAFT marker after it.\n"
        )
        result = parse_metadata_only(text)
        assert (
            result["additional_context"]
            == "This is the final section with no DRAFT marker after it."
        )


class TestBuildHandoffFromRawDraftAndMetadata:
    def test_combines_draft_and_metadata(self):
        draft_text = "This is the plain article body, no headers."
        handoff = build_handoff_from_raw_draft_and_metadata(
            draft_text, SAMPLE_METADATA, source_name="fallback"
        )
        assert handoff["draft"] == draft_text
        assert handoff["title"] == "How Fiber Reaches Rural Towns"
        assert (
            handoff["primary_claim"]
            == "Rural fiber buildouts are economically viable once subsidy math is right."
        )

    def test_title_falls_back_to_draft_h1_when_metadata_has_no_article_line(self):
        draft_text = "# Draft-Derived Title\n\nBody text."
        metadata_text = "PRIMARY CLAIM\nThe claim.\n"
        handoff = build_handoff_from_raw_draft_and_metadata(
            draft_text, metadata_text, source_name="fallback"
        )
        assert handoff["title"] == "Draft-Derived Title"

    def test_title_falls_back_to_source_name_without_article_or_h1(self):
        draft_text = "Just body text, no heading."
        metadata_text = "PRIMARY CLAIM\nThe claim.\n"
        handoff = build_handoff_from_raw_draft_and_metadata(
            draft_text, metadata_text, source_name="fallback-name"
        )
        assert handoff["title"] == "fallback-name"

    def test_metadata_article_line_wins_over_draft_h1(self):
        draft_text = "# Draft Title\n\nBody text."
        handoff = build_handoff_from_raw_draft_and_metadata(
            draft_text, SAMPLE_METADATA, source_name="fallback"
        )
        assert handoff["title"] == "How Fiber Reaches Rural Towns"
