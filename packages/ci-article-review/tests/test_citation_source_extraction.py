"""Tests for pipeline._extract_source_url — pulling a URL out of a fact-check
item's free-text "source" field (Pass 3 citation resolution)."""

from ci_article_review.pipeline import _extract_source_url


class TestExtractSourceUrl:
    def test_bare_url(self):
        assert (
            _extract_source_url("https://pubs.usgs.gov/publication/pp1894D")
            == "https://pubs.usgs.gov/publication/pp1894D"
        )

    def test_url_embedded_in_free_text(self):
        source = (
            "Constellation Energy, Press Release, "
            "https://www.constellationenergy.com/news/2024/deal"
        )
        assert (
            _extract_source_url(source)
            == "https://www.constellationenergy.com/news/2024/deal"
        )

    def test_strips_trailing_punctuation(self):
        source = "See https://example.com/report."
        assert _extract_source_url(source) == "https://example.com/report"

    def test_no_url_returns_none(self):
        assert _extract_source_url("Constellation Energy annual report, 2024") is None

    def test_empty_string_returns_none(self):
        assert _extract_source_url("") is None

    def test_none_returns_none(self):
        assert _extract_source_url(None) is None
