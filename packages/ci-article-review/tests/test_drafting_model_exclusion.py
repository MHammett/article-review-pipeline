"""The drafting model is kept out of the review domain it cannot judge.

``ai_speak.txt`` asks the reviewer to flag hedging, throat-clearing, vague
significance gesturing and the problem→cause→solution skeleton. Those are AI
defaults, so the model that wrote the draft is the worst available judge of
them — it under-reports its own habits. The pipeline therefore drops the
declared drafting model from ``voice_style``.

Declaration has two sources: ``pipeline.drafting_model`` in user.yaml as a
standing default, and a ``Drafted with:`` line in the handoff for one article.
"""

import logging

import pytest

from ci_article_review.handoff_parser import parse_draft_submission
from ci_article_review.pipeline import (
    _build_assignments,
    _drafter_is_excluded,
    _drafting_model,
)

ALL_KEYS = {
    m: {"api_key": "k"}
    for m in ("openai", "gemini", "mistral", "grok", "perplexity", "claude")
}
ALL_ENABLED = {m: {} for m in ALL_KEYS}


class TestDeclaringTheDraftingModel:
    def test_config_supplies_the_default(self):
        assert _drafting_model({}, {"drafting_model": "claude"}) == "claude"

    def test_handoff_overrides_the_config(self):
        """The drafting tool can change per article; the config cannot."""
        got = _drafting_model({"drafted_with": "openai"}, {"drafting_model": "claude"})
        assert got == "openai"

    def test_handoff_alone_is_enough(self):
        assert _drafting_model({"drafted_with": "gemini"}, {}) == "gemini"

    def test_undeclared_is_none(self):
        assert _drafting_model({}, {}) is None
        assert _drafting_model({"drafted_with": "  "}, {"drafting_model": ""}) is None

    def test_name_is_case_and_space_insensitive(self):
        assert _drafting_model({"drafted_with": "  Claude "}, {}) == "claude"

    def test_unknown_name_warns_and_excludes_nothing(self, caplog):
        """A typo costs a dropped review pass, not the run."""
        with caplog.at_level(logging.WARNING):
            assert _drafting_model({"drafted_with": "ChatGPT"}, {}) is None
        assert "ChatGPT" in caplog.text


class TestExclusionScope:
    def test_drafter_is_excluded_from_voice_style(self):
        assert _drafter_is_excluded("claude", "voice_style", "claude")

    @pytest.mark.parametrize(
        "domain",
        ["fact_check", "completeness", "argument_integrity", "red_team"],
    )
    def test_reasoning_domains_are_untouched(self, domain):
        """Only voice_style. The other four ask about the draft, not the prose."""
        assert not _drafter_is_excluded("claude", domain, "claude")

    def test_other_models_still_review_voice(self):
        assert not _drafter_is_excluded("openai", "voice_style", "claude")

    def test_no_declaration_excludes_nothing(self):
        assert not _drafter_is_excluded("claude", "voice_style", None)


class TestAssignments:
    def test_drafter_loses_only_its_voice_pass(self):
        pairs = _build_assignments("maximum", ALL_ENABLED, ALL_KEYS, "claude")
        assert ("claude", "voice_style") not in pairs
        assert ("claude", "fact_check") in pairs
        assert ("claude", "red_team") in pairs

    def test_the_other_models_still_cover_voice_style(self):
        pairs = _build_assignments("maximum", ALL_ENABLED, ALL_KEYS, "claude")
        reviewers = {m for m, d in pairs if d == "voice_style"}
        assert reviewers == {"openai", "gemini", "mistral", "grok", "perplexity"}

    def test_undeclared_drafter_changes_nothing(self):
        assert _build_assignments(
            "maximum", ALL_ENABLED, ALL_KEYS
        ) == _build_assignments("maximum", ALL_ENABLED, ALL_KEYS, None)

    def test_explicit_prompts_cannot_reintroduce_the_drafter(self):
        """`prompts: [voice_style]` is an override of the preset, not of this."""
        configs = dict(ALL_ENABLED, claude={"prompts": ["voice_style", "red_team"]})
        pairs = _build_assignments("standard", configs, ALL_KEYS, "claude")
        assert ("claude", "voice_style") not in pairs
        assert ("claude", "red_team") in pairs

    def test_warns_when_exclusion_leaves_voice_style_unreviewed(self, caplog):
        """At `standard`, voice_style is one model. If it drafted, nobody runs it.

        An empty voice section then means "never ran", which is indistinguishable
        in the report from "found nothing".
        """
        with caplog.at_level(logging.WARNING):
            pairs = _build_assignments("standard", ALL_ENABLED, ALL_KEYS, "openai")
        assert not [p for p in pairs if p[1] == "voice_style"]
        assert "voice_style" in caplog.text
        assert "never ran" in caplog.text or "not because" in caplog.text

    def test_no_warning_when_another_model_covers_it(self, caplog):
        with caplog.at_level(logging.WARNING):
            _build_assignments("maximum", ALL_ENABLED, ALL_KEYS, "claude")
        assert "No model is reviewing" not in caplog.text


class TestHandoffParsing:
    def _handoff(self, extra_line=""):
        return parse_draft_submission(
            "DRAFT SUBMISSION HANDOFF\n"
            "Article: A Real Title Here\n"
            "Publication: mikehammett\n"
            f"{extra_line}"
            "PRIMARY CLAIM\nA claim.\n\n"
            "DRAFT\nBody text.\n"
        )

    def test_drafted_with_is_parsed(self):
        assert self._handoff("Drafted with: claude\n")["drafted_with"] == "claude"

    def test_absent_line_yields_empty_string(self):
        assert self._handoff()["drafted_with"] == ""

    def test_absence_does_not_disturb_the_rest_of_the_handoff(self):
        h = self._handoff()
        assert h["title"] == "A Real Title Here"
        assert h["draft"].strip() == "Body text."
