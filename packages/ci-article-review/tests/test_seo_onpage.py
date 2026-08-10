"""Tests for the on-page checks in analysis.seo — images, links, title/H1,
meta-description length, and keyword usage.

Separate from test_seo.py, which covers the original structural checks and the
seo_rules plumbing. These are the checks added for authors who don't want to
learn SEO to find out an image has no alt text.
"""

from ci_article_review.analysis.seo import analyze, apply_suggestions, keyword_usage

_BODY = " ".join(["word"] * 320)
_ARTICLE = f"# A Title That Is Comfortably Long Enough\n\n{_BODY}"
_META = (
    "A description of the article long enough to work as a real search "
    "snippet, which means comfortably over seventy characters."
)
_HANDOFF = {
    "title": "A Title That Is Comfortably Long Enough",
    "seo": {"meta_description": _META},
}


def _types(result):
    return [i["type"] for i in result["issues"]]


class TestImageAltText:
    def test_markdown_image_without_alt_flagged(self):
        text = f"# Title Long Enough Here\n\n![](chart.png)\n\n{_BODY}"
        result = analyze(text, _HANDOFF)
        assert "missing_image_alt" in _types(result)

    def test_markdown_image_with_alt_not_flagged(self):
        text = (
            f"# Title Long Enough Here\n\n![Grid load by county](chart.png)\n\n{_BODY}"
        )
        assert "missing_image_alt" not in _types(analyze(text, _HANDOFF))

    def test_whitespace_alt_counts_as_missing(self):
        text = f"# Title Long Enough Here\n\n![   ](chart.png)\n\n{_BODY}"
        assert "missing_image_alt" in _types(analyze(text, _HANDOFF))

    def test_html_image_without_alt_flagged(self):
        # Embeds reach the draft as raw HTML and are preserved verbatim.
        text = f'# Title Long Enough Here\n\n<img src="chart.png">\n\n{_BODY}'
        assert "missing_image_alt" in _types(analyze(text, _HANDOFF))

    def test_html_image_with_alt_not_flagged(self):
        text = (
            f'# Title Long Enough Here\n\n<img src="c.png" alt="Grid load">\n\n{_BODY}'
        )
        assert "missing_image_alt" not in _types(analyze(text, _HANDOFF))

    def test_one_issue_summarizes_the_batch(self):
        images = "\n".join(f"![](img{i}.png)" for i in range(12))
        text = f"# Title Long Enough Here\n\n{images}\n\n{_BODY}"
        result = analyze(text, _HANDOFF)
        alt_issues = [i for i in result["issues"] if i["type"] == "missing_image_alt"]
        assert len(alt_issues) == 1
        assert "12 image(s)" in alt_issues[0]["detail"]


class TestAnchorText:
    def test_weak_anchor_text_flagged(self):
        text = (
            "# Title Long Enough Here\n\n"
            "See [click here](https://example.com/a) and [read more](https://example.com/b).\n\n"
            f"{_BODY}"
        )
        result = analyze(text, _HANDOFF)
        assert "weak_anchor_text" in _types(result)

    def test_descriptive_anchor_text_not_flagged(self):
        text = (
            "# Title Long Enough Here\n\n"
            "See [the county's zoning filing](https://example.com/a).\n\n"
            f"{_BODY}"
        )
        assert "weak_anchor_text" not in _types(analyze(text, _HANDOFF))

    def test_bare_url_as_anchor_text_flagged(self):
        text = (
            "# Title Long Enough Here\n\n"
            "[https://example.com/filing](https://example.com/filing)\n\n"
            f"{_BODY}"
        )
        assert "weak_anchor_text" in _types(analyze(text, _HANDOFF))

    def test_weak_word_inside_longer_text_not_flagged(self):
        # "here" alone is weak; "here is the filing" describes the destination.
        text = (
            "# Title Long Enough Here\n\n"
            "[here is the county filing](https://example.com/a)\n\n"
            f"{_BODY}"
        )
        assert "weak_anchor_text" not in _types(analyze(text, _HANDOFF))

    def test_image_syntax_is_not_read_as_a_link(self):
        text = f"# Title Long Enough Here\n\n![here](chart.png)\n\n{_BODY}"
        assert "weak_anchor_text" not in _types(analyze(text, _HANDOFF))


class TestInternalLinks:
    _SITE = "https://mysite.example"

    def test_no_internal_links_flagged_when_site_url_known(self):
        text = (
            "# Title Long Enough Here\n\n"
            "[the filing](https://elsewhere.example/a)\n\n"
            f"{_BODY}"
        )
        result = analyze(text, _HANDOFF, site_url=self._SITE)
        assert "no_internal_links" in _types(result)

    def test_absolute_internal_link_satisfies_the_check(self):
        text = (
            "# Title Long Enough Here\n\n"
            "[earlier coverage](https://mysite.example/prior-post)\n\n"
            f"{_BODY}"
        )
        assert "no_internal_links" not in _types(
            analyze(text, _HANDOFF, site_url=self._SITE)
        )

    def test_www_prefix_still_matches(self):
        text = (
            "# Title Long Enough Here\n\n"
            "[earlier coverage](https://www.mysite.example/prior)\n\n"
            f"{_BODY}"
        )
        assert "no_internal_links" not in _types(
            analyze(text, _HANDOFF, site_url=self._SITE)
        )

    def test_relative_link_counts_as_internal(self):
        text = f"# Title Long Enough Here\n\n[earlier](/prior-post)\n\n{_BODY}"
        assert "no_internal_links" not in _types(
            analyze(text, _HANDOFF, site_url=self._SITE)
        )

    def test_skipped_entirely_without_a_site_url(self):
        # Nothing to compare against, so the check is skipped rather than guessed.
        text = (
            "# Title Long Enough Here\n\n"
            "[the filing](https://elsewhere.example/a)\n\n"
            f"{_BODY}"
        )
        assert "no_internal_links" not in _types(analyze(text, _HANDOFF))

    def test_no_links_at_all_is_not_an_internal_link_finding(self):
        assert "no_internal_links" not in _types(
            analyze(_ARTICLE, _HANDOFF, site_url=self._SITE)
        )


class TestTitleH1Mismatch:
    def test_matching_title_and_h1_not_flagged(self):
        assert "title_h1_mismatch" not in _types(analyze(_ARTICLE, _HANDOFF))

    def test_differing_title_and_h1_flagged(self):
        handoff = {**_HANDOFF, "title": "A Completely Different Search Title"}
        result = analyze(_ARTICLE, handoff)
        assert "title_h1_mismatch" in _types(result)

    def test_case_and_spacing_differences_are_not_a_mismatch(self):
        handoff = {**_HANDOFF, "title": "a title that is  COMFORTABLY long enough"}
        assert "title_h1_mismatch" not in _types(analyze(_ARTICLE, handoff))

    def test_no_h1_is_not_reported_as_a_mismatch(self):
        text = f"No heading at all here.\n\n{_BODY}"
        assert "title_h1_mismatch" not in _types(analyze(text, _HANDOFF))


class TestSuppliedMetaDescriptionLength:
    def test_over_long_description_flagged(self):
        handoff = {**_HANDOFF, "seo": {"meta_description": "x" * 200}}
        assert "meta_description_too_long" in _types(analyze(_ARTICLE, handoff))

    def test_too_short_description_flagged(self):
        handoff = {**_HANDOFF, "seo": {"meta_description": "Too short."}}
        assert "meta_description_too_short" in _types(analyze(_ARTICLE, handoff))

    def test_usable_description_not_flagged(self):
        types = _types(analyze(_ARTICLE, _HANDOFF))
        assert "meta_description_too_long" not in types
        assert "meta_description_too_short" not in types

    def test_limits_come_from_seo_rules(self):
        handoff = {**_HANDOFF, "seo": {"meta_description": "x" * 100}}
        result = analyze(
            _ARTICLE, handoff, seo_rules={"meta_description_max_chars": 90}
        )
        assert "meta_description_too_long" in _types(result)

    def test_absent_description_reports_absence_not_length(self):
        types = _types(analyze(_ARTICLE, {**_HANDOFF, "seo": {}}))
        assert "no_meta_description" in types
        assert "meta_description_too_short" not in types


class TestKeywordUsage:
    _TEXT = (
        "# Interconnection Queues Decide the Timeline\n\n"
        "The interconnection queue is the constraint, not generation.\n\n"
        "## How the interconnection queue works\n\n"
        "Some body text about the interconnection queue and other things."
    )

    def test_phrase_found_everywhere_reports_every_placement(self):
        usage = keyword_usage(
            self._TEXT,
            "Interconnection Queues Decide the Timeline",
            "interconnection queue",
        )
        assert usage["in_title"] is True
        assert usage["in_opening"] is True
        assert len(usage["in_headings"]) == 2
        assert usage["body_count"] >= 4

    def test_absent_phrase_reports_zero(self):
        usage = keyword_usage(self._TEXT, "A Title", "solar curtailment")
        assert usage["body_count"] == 0
        assert usage["in_title"] is False
        assert usage["in_opening"] is False
        assert usage["in_headings"] == []

    def test_matching_is_case_and_whitespace_insensitive(self):
        usage = keyword_usage(self._TEXT, "A Title", "  INTERCONNECTION   QUEUE ")
        assert usage["body_count"] >= 4

    def test_opening_excludes_the_h1(self):
        text = "# Solar Curtailment Explained\n\nThe opening says nothing about it."
        usage = keyword_usage(text, "Solar Curtailment Explained", "solar curtailment")
        assert usage["in_title"] is True
        assert usage["in_opening"] is False

    def test_empty_keyword_is_not_reported_as_present(self):
        usage = keyword_usage(self._TEXT, "A Title", "")
        assert usage["body_count"] == 0
        assert usage["in_title"] is False

    def test_apply_suggestions_annotates_each_candidate(self):
        result = analyze(
            self._TEXT, {"title": "Interconnection Queues Decide the Timeline"}
        )
        suggestions = {
            "status": "ok",
            "keyword_candidates": [
                {"keyword": "interconnection queue", "rationale": "a"},
                {"keyword": "solar curtailment", "rationale": "b"},
            ],
            "fields": {},
        }
        apply_suggestions(
            result,
            suggestions,
            text=self._TEXT,
            title="Interconnection Queues Decide the Timeline",
        )

        used, unused = suggestions["keyword_candidates"]
        assert used["usage"]["body_count"] >= 4
        # The finding the whole feature exists to surface.
        assert unused["usage"]["body_count"] == 0

    def test_apply_suggestions_without_text_skips_the_scan(self):
        suggestions = {
            "status": "ok",
            "keyword_candidates": [{"keyword": "anything", "rationale": ""}],
            "fields": {},
        }
        apply_suggestions(analyze(self._TEXT, {}), suggestions)
        assert "usage" not in suggestions["keyword_candidates"][0]
