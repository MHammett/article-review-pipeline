"""Tests for analysis.seo."""

from ci_article_review.analysis.seo import (
    DRAFT_MODE,
    PUBLISH_MODE,
    analyze,
    apply_suggestions,
    meta_description_limit,
    suggestions_enabled,
    title_limit,
)

_GOOD_ARTICLE = "\n".join(
    [
        "# A Well-Constructed Article About Important Things",
        "",
        "## Introduction",
        "",
        " ".join(["word"] * 320),  # 320 words, meets minimum
    ]
)

_HANDOFF_WITH_META = {
    "title": "A Well-Constructed Article About Important Things",
    "seo": {"meta_description": "A description under 160 characters."},
}


class TestSeoAnalyze:
    def test_returns_all_keys(self):
        result = analyze(_GOOD_ARTICLE, _HANDOFF_WITH_META)
        for k in (
            "title",
            "title_length",
            "h1_count",
            "h2_count",
            "h3_count",
            "word_count",
            "has_meta_description",
            "issues",
        ):
            assert k in result, f"missing: {k}"

    def test_good_article_no_issues(self):
        result = analyze(_GOOD_ARTICLE, _HANDOFF_WITH_META)
        assert result["issues"] == [], result["issues"]

    def test_title_too_long_flagged(self):
        handoff = {
            "title": "A" * 61,
            "seo": {"meta_description": "fine"},
        }
        result = analyze(_GOOD_ARTICLE, handoff)
        types = [i["type"] for i in result["issues"]]
        assert "title_too_long" in types

    def test_no_h1_flagged(self):
        no_h1 = "## Just a H2\n\n" + " ".join(["word"] * 320)
        result = analyze(no_h1, _HANDOFF_WITH_META)
        types = [i["type"] for i in result["issues"]]
        assert "no_h1" in types

    def test_multiple_h1_flagged(self):
        two_h1 = "# First\n\n# Second\n\n" + " ".join(["word"] * 320)
        result = analyze(two_h1, _HANDOFF_WITH_META)
        types = [i["type"] for i in result["issues"]]
        assert "multiple_h1" in types

    def test_thin_content_flagged(self):
        short = "# Title\n\nToo short."
        result = analyze(short, _HANDOFF_WITH_META)
        types = [i["type"] for i in result["issues"]]
        assert "thin_content" in types

    def test_no_meta_description_flagged(self):
        result = analyze(_GOOD_ARTICLE, {"title": "Fine", "seo": {}})
        types = [i["type"] for i in result["issues"]]
        assert "no_meta_description" in types

    def test_heading_hierarchy_h3_without_h2(self):
        text = "# Title\n\n### Jump to H3\n\n" + " ".join(["word"] * 320)
        result = analyze(text, _HANDOFF_WITH_META)
        types = [i["type"] for i in result["issues"]]
        assert "heading_hierarchy" in types

    def test_none_handoff_no_crash(self):
        result = analyze(_GOOD_ARTICLE, None)
        assert isinstance(result["issues"], list)

    def test_h1_falls_back_to_first_heading(self):
        text = "# A Detected Title Here That Is Long Enough\n\n" + " ".join(
            ["word"] * 320
        )
        result = analyze(text, {"seo": {"meta_description": "ok"}})
        assert result["title"] == "A Detected Title Here That Is Long Enough"


class TestSeoRules:
    def test_custom_title_max_flags_shorter_title(self):
        # A 40-char title passes the default 60 but fails a custom 30 ceiling.
        handoff = {"title": "A" * 40, "seo": {"meta_description": "fine"}}
        result = analyze(_GOOD_ARTICLE, handoff, seo_rules={"title_max_chars": 30})
        assert "title_too_long" in [i["type"] for i in result["issues"]]

    def test_custom_min_words_flags_longer_article(self):
        # 320 words passes the default 300 but fails a custom 500 minimum.
        result = analyze(
            _GOOD_ARTICLE, _HANDOFF_WITH_META, seo_rules={"min_article_words": 500}
        )
        assert "thin_content" in [i["type"] for i in result["issues"]]

    def test_custom_title_min(self):
        handoff = {
            "title": "Short title here",
            "seo": {"meta_description": "fine"},
        }  # 16 chars
        result = analyze(_GOOD_ARTICLE, handoff, seo_rules={"title_min_chars": 30})
        assert "title_too_short" in [i["type"] for i in result["issues"]]

    def test_non_integer_rule_falls_back_to_default(self):
        # A typo must not crash the pre-analysis pass — it should use the default (60).
        handoff = {"title": "A" * 40, "seo": {"meta_description": "fine"}}
        result = analyze(_GOOD_ARTICLE, handoff, seo_rules={"title_max_chars": "sixty"})
        # 40 chars is fine against the default 60 → no title_too_long
        assert "title_too_long" not in [i["type"] for i in result["issues"]]

    def test_zero_or_negative_rule_falls_back_to_default(self):
        result = analyze(
            _GOOD_ARTICLE, _HANDOFF_WITH_META, seo_rules={"min_article_words": 0}
        )
        # 0 is invalid → default 300 → 320-word article is not thin
        assert "thin_content" not in [i["type"] for i in result["issues"]]

    def test_none_seo_rules_uses_defaults(self):
        result = analyze(_GOOD_ARTICLE, _HANDOFF_WITH_META, seo_rules=None)
        assert result["issues"] == []

    def test_meta_description_limit_default_and_override(self):
        assert meta_description_limit() == 155
        assert meta_description_limit({"meta_description_max_chars": 120}) == 120
        # Same bad-value contract as the other rules.
        assert meta_description_limit({"meta_description_max_chars": "long"}) == 155
        assert meta_description_limit({"meta_description_max_chars": 0}) == 155

    def test_title_limit_default_and_override(self):
        assert title_limit() == 60
        assert title_limit({"title_max_chars": 55}) == 55

    def test_suggestions_enabled_defaults_on(self):
        assert suggestions_enabled() is True
        assert suggestions_enabled({}) is True
        assert suggestions_enabled({"suggestions": False}) is False
        # A non-boolean must not silently disable the pass.
        assert suggestions_enabled({"suggestions": "nope"}) is True


def _meta_issue(result):
    return next(i for i in result["issues"] if i["type"] == "no_meta_description")


class TestNoMetaDescriptionIsActionable:
    """Template A has no SEO section, so the draft-mode finding cannot say the
    author left one out of a section their document has no way to contain."""

    def test_draft_mode_does_not_point_at_a_section_the_template_lacks(self):
        result = analyze(_GOOD_ARTICLE, {"title": "Fine", "seo": {}})
        detail = _meta_issue(result)["detail"]

        assert detail != "No meta description in handoff SEO METADATA section"
        assert "Template A" in detail
        assert "Template C" in detail

    def test_publish_mode_keeps_the_original_wording(self):
        result = analyze(_GOOD_ARTICLE, {"title": "Fine", "seo": {}}, mode=PUBLISH_MODE)
        assert (
            _meta_issue(result)["detail"]
            == "No meta description in handoff SEO METADATA section"
        )

    def test_mode_is_recorded_and_defaults_to_draft(self):
        assert analyze(_GOOD_ARTICLE, _HANDOFF_WITH_META)["mode"] == DRAFT_MODE
        assert (
            analyze(_GOOD_ARTICLE, _HANDOFF_WITH_META, mode=PUBLISH_MODE)["mode"]
            == PUBLISH_MODE
        )


class TestApplySuggestions:
    _SUGGESTIONS = {
        "status": "ok",
        "keyword_candidates": [{"keyword": "a phrase", "rationale": "why"}],
        "meta_description": "A drafted description.",
        "meta_description_chars": 22,
        "meta_description_limit": 155,
        "meta_description_over_limit": False,
    }

    def test_attaches_the_block(self):
        result = analyze(_GOOD_ARTICLE, {"title": "Fine", "seo": {}})
        apply_suggestions(result, self._SUGGESTIONS)
        assert result["suggestions"] is self._SUGGESTIONS

    def test_points_the_finding_at_the_suggestion(self):
        result = analyze(_GOOD_ARTICLE, {"title": "Fine", "seo": {}})
        apply_suggestions(result, self._SUGGESTIONS)

        detail = _meta_issue(result)["detail"]
        assert "suggested draft" in detail
        # Still the same finding type — the issue count trend stays comparable
        # across runs (see history_analytics._quality_metrics).
        assert _meta_issue(result)["type"] == "no_meta_description"

    def test_publish_mode_finding_also_points_at_the_suggestion(self):
        result = analyze(_GOOD_ARTICLE, {"title": "Fine", "seo": {}}, mode=PUBLISH_MODE)
        apply_suggestions(result, self._SUGGESTIONS)
        assert "suggested draft" in _meta_issue(result)["detail"]

    def test_unavailable_suggestions_leave_the_standalone_wording(self):
        result = analyze(_GOOD_ARTICLE, {"title": "Fine", "seo": {}})
        before = _meta_issue(result)["detail"]
        apply_suggestions(result, {"status": "failed", "reason": "call failed"})

        # The reason is still attached so the run says why, but the finding
        # must not promise a draft that isn't there.
        assert result["suggestions"]["status"] == "failed"
        assert _meta_issue(result)["detail"] == before
        assert "suggested draft" not in _meta_issue(result)["detail"]

    def test_no_suggestions_at_all_is_a_no_op(self):
        result = analyze(_GOOD_ARTICLE, {"title": "Fine", "seo": {}})
        apply_suggestions(result, None)
        assert "suggestions" not in result

    def test_supplied_meta_description_means_no_finding_to_rewrite(self):
        result = analyze(_GOOD_ARTICLE, _HANDOFF_WITH_META)
        apply_suggestions(result, self._SUGGESTIONS)
        assert result["issues"] == []
        assert result["suggestions"] is self._SUGGESTIONS
