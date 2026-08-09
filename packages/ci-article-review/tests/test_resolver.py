"""Tests for adapters.citation.resolver — parallel resolution, ordering, pointer flag."""

from unittest.mock import patch

from ci_article_review.adapters.citation import resolver


_SOURCES = [{"name": "FRED", "adapter": "fred"}]


def _no_wayback(url, timeout=10):
    return {"archived": None}


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


class TestArchiveSubmission:
    def _archived_wayback(self, url, timeout=10):
        return {"archived": True}

    def _unarchived_wayback(self, url, timeout=10):
        return {"archived": False}

    def test_submits_only_when_not_archived(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=self._unarchived_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
                return_value={"submitted": True, "job_id": None},
            ) as mock_submit,
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        mock_submit.assert_called_once()
        assert mock_submit.call_args.args[0] == "https://x"
        assert results[0]["wayback"]["submitted"] is True

    def test_does_not_submit_when_already_archived(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=self._archived_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
            ) as mock_submit,
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        mock_submit.assert_not_called()
        assert "submitted" not in results[0]["wayback"]

    def test_submission_failure_does_not_raise_or_fail_resolution(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=self._unarchived_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
                side_effect=RuntimeError("rate limited"),
            ),
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        assert results[0]["resolved"] is True
        assert results[0]["wayback"]["submitted"] is False
        assert "rate limited" in results[0]["wayback"]["submission_error"]

    def test_archive_org_credentials_passed_through(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=self._unarchived_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
                return_value={"submitted": True, "job_id": "j1"},
            ) as mock_submit,
        ):
            resolver.resolve_citations(
                ["c"],
                _SOURCES,
                api_keys={"archive_org": {"access_key": "AK", "secret_key": "SK"}},
            )

        assert mock_submit.call_args.kwargs["access_key"] == "AK"
        assert mock_submit.call_args.kwargs["secret_key"] == "SK"

    def test_no_submission_when_unresolved(self):
        def fake_resolve(claim, api_key=None):
            return {"found": False}

        with (
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
            ) as mock_submit,
        ):
            resolver.resolve_citations(["c"], _SOURCES)

        mock_submit.assert_not_called()
