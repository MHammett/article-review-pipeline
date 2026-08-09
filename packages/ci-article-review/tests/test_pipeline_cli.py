"""Tests for the --raw-draft / --metadata CLI wiring in pipeline.main().

Added alongside commit 803440c but previously untested. Follows the
patch-and-assert pattern used in test_webpage.py's TestUrlModeFlowsIntoReview.
"""

import sys
from unittest.mock import patch

import pytest


class TestRawDraftArgparse:
    def test_raw_draft_mutually_exclusive_with_draft(self):
        import ci_article_review.pipeline as pipeline

        argv = [
            "pipeline.py",
            "--draft",
            "handoff.md",
            "--raw-draft",
            "draft.md",
            "--publication",
            "myblog",
        ]
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                pipeline.main()

    def test_raw_draft_mutually_exclusive_with_publish(self):
        import ci_article_review.pipeline as pipeline

        argv = [
            "pipeline.py",
            "--publish",
            "handoff.md",
            "--raw-draft",
            "draft.md",
            "--publication",
            "myblog",
        ]
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                pipeline.main()

    def test_metadata_without_raw_draft_errors(self):
        import ci_article_review.pipeline as pipeline

        argv = [
            "pipeline.py",
            "--metadata",
            "meta.md",
            "--publication",
            "myblog",
        ]
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                pipeline.main()


class TestRawDraftModeFlowsIntoReview:
    """--raw-draft alone must build a raw-text handoff and reach run_draft_pipeline."""

    def test_raw_draft_alone_builds_handoff(self):
        import ci_article_review.pipeline as pipeline

        argv = [
            "pipeline.py",
            "--raw-draft",
            "draft.md",
            "--publication",
            "myblog",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch(
                "ci_article_review.pipeline._read_handoff_file",
                return_value="# My Draft Title\n\nSome article body text.",
            ) as mock_read,
            patch("ci_article_review.pipeline.logging.FileHandler"),
            patch("logging.Logger.addHandler"),
            patch("ci_article_review.pipeline.run_draft_pipeline") as mock_run,
        ):
            pipeline.main()

        mock_read.assert_called_once_with("draft.md")
        mock_run.assert_called_once()
        handoff = mock_run.call_args.kwargs["handoff"]
        assert handoff["title"] == "My Draft Title"
        assert "Some article body text." in handoff["draft"]
        assert handoff["run_number"] == 1
        # File-path argument is None — the pipeline uses the pre-built handoff.
        assert mock_run.call_args.args[0] is None

    def test_raw_draft_falls_back_to_filename_stem_for_title(self):
        import ci_article_review.pipeline as pipeline

        argv = [
            "pipeline.py",
            "--raw-draft",
            "/tmp/my-cool-draft.md",
            "--publication",
            "myblog",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch(
                "ci_article_review.pipeline._read_handoff_file",
                return_value="Body text with no H1 heading.",
            ),
            patch("ci_article_review.pipeline.logging.FileHandler"),
            patch("logging.Logger.addHandler"),
            patch("ci_article_review.pipeline.run_draft_pipeline") as mock_run,
        ):
            pipeline.main()

        handoff = mock_run.call_args.kwargs["handoff"]
        assert handoff["title"] == "my-cool-draft"


class TestRawDraftWithMetadataFlowsIntoReview:
    """--raw-draft + --metadata must combine both files into one handoff."""

    def test_combines_draft_and_metadata_files(self):
        import ci_article_review.pipeline as pipeline

        argv = [
            "pipeline.py",
            "--raw-draft",
            "draft.md",
            "--metadata",
            "meta.md",
            "--publication",
            "myblog",
        ]
        draft_text = "Plain article body, no headers."
        metadata_text = (
            "Article: Metadata-Supplied Title\n\n"
            "PRIMARY CLAIM\nThe claim from metadata.\n\n"
            "TARGET AUDIENCE\nEveryone.\n"
        )

        def fake_read(path):
            return {"draft.md": draft_text, "meta.md": metadata_text}[path]

        with (
            patch.object(sys, "argv", argv),
            patch(
                "ci_article_review.pipeline._read_handoff_file",
                side_effect=fake_read,
            ) as mock_read,
            patch("ci_article_review.pipeline.logging.FileHandler"),
            patch("logging.Logger.addHandler"),
            patch("ci_article_review.pipeline.run_draft_pipeline") as mock_run,
        ):
            pipeline.main()

        assert mock_read.call_count == 2
        handoff = mock_run.call_args.kwargs["handoff"]
        assert handoff["title"] == "Metadata-Supplied Title"
        assert handoff["draft"] == draft_text
        assert handoff["primary_claim"] == "The claim from metadata."
        assert handoff["target_audience"] == "Everyone."
        assert mock_run.call_args.args[0] is None


class TestBuildUserPromptForwardsMetadataFields:
    """_build_user_prompt must include target_audience/sources_cited/uncertain_sections/known_gaps
    when present, rather than silently dropping them (fixed alongside 803440c)."""

    def test_all_optional_fields_forwarded(self):
        from ci_article_review.pipeline import _build_user_prompt

        handoff = {
            "title": "A Title",
            "target_audience": "Municipal officials.",
            "primary_claim": "The claim.",
            "sources_cited": "FCC broadband map.",
            "uncertain_sections": "Paragraph 4 figures.",
            "known_gaps": "No satellite discussion.",
        }
        prompt = _build_user_prompt("draft body", handoff)
        assert "TARGET AUDIENCE: Municipal officials." in prompt
        assert "SOURCES ALREADY CITED:\nFCC broadband map." in prompt
        assert "UNCERTAIN SECTIONS" in prompt and "Paragraph 4 figures." in prompt
        assert "KNOWN GAPS" in prompt and "No satellite discussion." in prompt

    def test_missing_optional_fields_omitted(self):
        from ci_article_review.pipeline import _build_user_prompt

        handoff = {"title": "A Title"}
        prompt = _build_user_prompt("draft body", handoff)
        assert "TARGET AUDIENCE" not in prompt
        assert "SOURCES ALREADY CITED" not in prompt
        assert "UNCERTAIN SECTIONS" not in prompt
        assert "KNOWN GAPS" not in prompt
