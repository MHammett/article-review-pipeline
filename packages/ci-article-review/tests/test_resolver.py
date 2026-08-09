"""Tests for adapters.citation.resolver — parallel resolution, ordering, pointer flag."""

import json
from unittest.mock import patch

from ci_article_review.adapters.citation import resolver


_SOURCES = [{"name": "FRED", "adapter": "fred"}]


def _no_wayback(url, timeout=10):
    return {"archived": None}


def _write_report(root, slug, run_number, ts, citations):
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"run_{run_number}_{ts.replace(':', '').replace('-', '')}_report.json"
    report = {
        "generated": ts,
        "run_number": run_number,
        "article_title": slug,
        "section_9_citations": citations,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return path


class TestResolveCitations:
    def test_empty_claims(self):
        assert resolver.resolve_citations([], _SOURCES) == []

    def test_preserves_claim_order(self):
        claims = ["claim zero", "claim one", "claim two"]

        def fake_resolve(claim, api_key=None):
            return {
                "found": True,
                "url": f"https://x/{claim.split()[-1]}",
                "content": claim,
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(claims, _SOURCES)

        assert [r["claim"] for r in results] == claims

    def test_unresolved_claim_marked(self):
        def fake_resolve(claim, api_key=None):
            return {"found": False}

        with patch(
            "ci_article_review.adapters.citation.sources.fred.resolve",
            side_effect=fake_resolve,
        ):
            results = resolver.resolve_citations(["unknown claim"], _SOURCES)

        assert results[0]["resolved"] is False
        assert "note" in results[0]

    def test_checksum_verification_label(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        assert results[0]["verification"] == "checksum"

    def test_pointer_only_label(self):
        def fake_resolve(claim, api_key=None):
            return {
                "found": True,
                "pointer_only": True,
                "url": "https://x",
                "content": "ptr",
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        assert results[0]["verification"] == "pointer"

    def test_missing_source_name_does_not_crash(self):
        sources = [{"adapter": "fred"}]  # no "name" key

        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(["c"], sources)

        # Falls back to adapter name instead of raising KeyError
        assert results[0]["source_name"] == "fred"

    def test_adapter_exception_is_isolated(self):
        def boom(claim, api_key=None):
            raise RuntimeError("adapter exploded")

        with patch(
            "ci_article_review.adapters.citation.sources.fred.resolve", side_effect=boom
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        # Exception is caught; claim reported unresolved rather than crashing the run
        assert results[0]["resolved"] is False

    def test_known_url_resolves_without_adapter_loop(self):
        """A claim carrying a known_url (e.g. supplied by the fact-check model)
        must be fetched and checksummed directly, never touching the adapter loop.
        """
        mock_resp = type(
            "R", (), {"raise_for_status": lambda self: None, "text": "page content"}
        )()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ) as mock_get,
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve"
            ) as mock_adapter,
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is True
        assert results[0]["url"] == "https://example.com/page"
        assert results[0]["source_name"] == "fact-check model"
        assert results[0]["verification"] == "checksum"
        mock_get.assert_called_once()
        mock_adapter.assert_not_called()

    def test_known_url_fetch_failure_reports_unresolved(self):
        with patch(
            "ci_article_review.adapters.citation.resolver.requests.get",
            side_effect=RuntimeError("timeout"),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/broken"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is False
        assert "note" in results[0]

    def test_dict_entry_without_known_url_uses_adapter_loop(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": None}], _SOURCES
            )

        assert results[0]["resolved"] is True
        assert results[0]["source_name"] == "FRED"


class TestContentDriftDetection:
    """Cross-run checksum comparison: a URL resolved before, with a different
    checksum now, should be flagged as content_changed_since."""

    def test_same_url_same_checksum_no_drift(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [
                {
                    "claim": "old claim",
                    "url": "https://example.com/data",
                    "checksum": resolver.sha256_checksum("same content"),
                    "resolved": True,
                }
            ],
        )

        def fake_resolve(claim, api_key=None):
            return {
                "found": True,
                "url": "https://example.com/data",
                "content": "same content",
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(
                ["new claim"], _SOURCES, history_root=str(tmp_path)
            )

        assert "content_changed_since" not in results[0]

    def test_same_url_different_checksum_flags_drift(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            3,
            "2026-01-01T00:00:00",
            [
                {
                    "claim": "old claim",
                    "url": "https://example.com/data",
                    "checksum": resolver.sha256_checksum("old content"),
                    "resolved": True,
                }
            ],
        )

        def fake_resolve(claim, api_key=None):
            return {
                "found": True,
                "url": "https://example.com/data",
                "content": "new content",
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(
                ["new claim"], _SOURCES, history_root=str(tmp_path)
            )

        drift = results[0]["content_changed_since"]
        assert drift["prior_run"] == 3
        assert drift["prior_article"] == "article-one"
        assert drift["prior_date"] == "2026-01-01T00:00:00"
        assert drift["prior_checksum"] == resolver.sha256_checksum("old content")

    def test_never_before_seen_url_no_comparison(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [
                {
                    "claim": "old claim",
                    "url": "https://example.com/unrelated",
                    "checksum": resolver.sha256_checksum("whatever"),
                    "resolved": True,
                }
            ],
        )

        def fake_resolve(claim, api_key=None):
            return {
                "found": True,
                "url": "https://example.com/brand-new",
                "content": "brand new content",
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(
                ["new claim"], _SOURCES, history_root=str(tmp_path)
            )

        assert results[0]["resolved"] is True
        assert "content_changed_since" not in results[0]

    def test_drift_detected_across_different_articles(self, tmp_path):
        """The same source URL cited by a different article's prior run still
        counts — sources get reused across articles for the same publication."""
        _write_report(
            tmp_path,
            "some-other-article",
            1,
            "2026-01-01T00:00:00",
            [
                {
                    "claim": "unrelated claim",
                    "url": "https://example.com/shared-source",
                    "checksum": resolver.sha256_checksum("v1"),
                    "resolved": True,
                }
            ],
        )

        def fake_resolve(claim, api_key=None):
            return {
                "found": True,
                "url": "https://example.com/shared-source",
                "content": "v2",
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(
                ["new claim"], _SOURCES, history_root=str(tmp_path)
            )

        assert results[0]["content_changed_since"]["prior_article"] == (
            "some-other-article"
        )

    def test_empty_history_root_no_crash(self, tmp_path):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(
                ["c"], _SOURCES, history_root=str(tmp_path / "does-not-exist")
            )

        assert "content_changed_since" not in results[0]
