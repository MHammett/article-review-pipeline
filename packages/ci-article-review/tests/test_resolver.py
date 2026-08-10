"""Tests for adapters.citation.resolver — parallel resolution, ordering, pointer flag."""

import json
from unittest.mock import MagicMock, patch

import requests

from ci_article_review.adapters.citation import resolver


_SOURCES = [{"name": "FRED", "adapter": "fred"}]


def _no_wayback(url, timeout=10):
    return {"archived": None}


def _prior_citation(url, content, verification="checksum"):
    """A section_9_citations entry as a prior run would have saved it."""
    return {
        "claim": "old claim",
        "url": url,
        "checksum": resolver.sha256_checksum(content),
        "resolved": True,
        "verification": verification,
    }


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

    def test_no_api_keys_skips_relevance_check(self):
        """Without a mistral API key, the relevance check is skipped and the
        citation degrades to the pre-existing (unverified-relevance) behavior
        rather than blocking resolution."""
        mock_resp = type(
            "R", (), {"raise_for_status": lambda self: None, "text": "page content"}
        )()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call"
            ) as mock_mistral,
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
            )

        mock_mistral.assert_not_called()
        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "checksum"
        assert "relevance_check" in results[0]

    def test_relevance_check_supports_claim_stays_checksum_verified(self):
        mock_resp = type(
            "R", (), {"raise_for_status": lambda self: None, "text": "page content"}
        )()
        call_log = []

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
                return_value={
                    "failed": False,
                    "data": {"verdict": "supports", "reason": "matches"},
                    "model": "mistral-small-latest",
                    "tokens": {"prompt": 10, "completion": 5},
                    "elapsed_seconds": 0.2,
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
                verification_call_log=call_log,
            )

        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "checksum"
        assert results[0]["relevance_verdict"] == "supports"
        assert len(call_log) == 1
        assert call_log[0]["failed"] is False
        assert call_log[0]["model"] == "mistral-small-latest"

    def test_relevance_check_contradicts_downgrades_citation(self):
        mock_resp = type(
            "R", (), {"raise_for_status": lambda self: None, "text": "page content"}
        )()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
                return_value={
                    "failed": False,
                    "data": {
                        "verdict": "not_addressed",
                        "reason": "page never mentions this",
                    },
                    "model": "mistral-small-latest",
                    "tokens": {"prompt": 10, "completion": 5},
                    "elapsed_seconds": 0.2,
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
            )

        assert results[0]["resolved"] is False
        assert results[0]["verification"] != "checksum"
        assert "note" in results[0]

    def test_relevance_check_call_failure_degrades_gracefully(self):
        """The verification call itself failing (rate limit, timeout, etc.)
        must not crash resolution or block the citation — it degrades back
        to the pre-existing unverified-relevance behavior with a note."""
        mock_resp = type(
            "R", (), {"raise_for_status": lambda self: None, "text": "page content"}
        )()
        call_log = []

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
                return_value={
                    "failed": True,
                    "error": "rate limited",
                    "model": "mistral-small-latest",
                    "tokens": {},
                    "elapsed_seconds": 0.1,
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
                verification_call_log=call_log,
            )

        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "checksum"
        assert "relevance_check" in results[0]
        assert len(call_log) == 1
        assert call_log[0]["failed"] is True

    def test_relevance_check_exception_does_not_crash_resolution(self):
        mock_resp = type(
            "R", (), {"raise_for_status": lambda self: None, "text": "page content"}
        )()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
                side_effect=RuntimeError("boom"),
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
            )

        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "checksum"
        assert "relevance_check" in results[0]

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


def _http_error_response(status_code):
    resp = MagicMock()
    resp.status_code = status_code
    error = requests.exceptions.HTTPError(f"{status_code} error", response=resp)
    resp.raise_for_status.side_effect = error
    return resp


class TestKnownUrlWaybackFallback:
    """A direct 403 on a known_url should fall back to a Wayback snapshot;
    404/5xx should not — see resolver._wayback_fallback_content scoping."""

    def test_403_recovers_via_wayback_snapshot(self):
        snapshot_url = (
            "https://web.archive.org/web/20240101000000/https://example.com/page"
        )
        snap_resp = MagicMock(text="archived page content")
        snap_resp.raise_for_status.return_value = None

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                side_effect=[_http_error_response(403), snap_resp],
            ) as mock_get,
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                return_value={"archived": True, "snapshot_url": snapshot_url},
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is True
        assert results[0]["verified_via"] == "wayback_fallback"
        assert results[0]["checksum"] == resolver.sha256_checksum(
            "archived page content"
        )
        assert mock_get.call_count == 2

    def test_403_with_no_snapshot_reports_unresolved(self):
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=_http_error_response(403),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                return_value={"archived": False},
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/blocked"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is False
        assert "note" in results[0]

    def test_404_does_not_attempt_wayback_fallback(self):
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=_http_error_response(404),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check"
            ) as mock_wb,
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/gone"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is False
        mock_wb.assert_not_called()


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


def _fred_returning(url, content, pointer_only=False):
    def fake_resolve(claim, api_key=None):
        result = {"found": True, "url": url, "content": content}
        if pointer_only:
            result["pointer_only"] = True
        return result

    return fake_resolve


def _resolve_against_history(history_root, fake_resolve, claims=None):
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
        return resolver.resolve_citations(
            claims or ["new claim"], _SOURCES, history_root=str(history_root)
        )


class TestContentDriftDetection:
    """Cross-run checksum comparison: a URL resolved at the checksum tier
    before, with a different checksum now, is flagged content_changed_since."""

    def test_same_url_same_checksum_no_drift(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "same content")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/data", "same content")
        )

        assert results[0]["verification"] == "checksum"
        assert "content_changed_since" not in results[0]

    def test_same_url_different_checksum_flags_drift(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            3,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "old content")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/data", "new content")
        )

        drift = results[0]["content_changed_since"]
        assert drift["prior_run"] == 3
        assert drift["prior_article"] == "article-one"
        assert drift["prior_date"] == "2026-01-01T00:00:00"
        assert drift["prior_checksum"] == resolver.sha256_checksum("old content")

    def test_drift_never_blocks_resolution(self, tmp_path):
        """A changed source is a reviewer signal, not a resolution failure."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "old content")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/data", "new content")
        )

        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "checksum"
        assert "content_changed_since" in results[0]

    def test_never_before_seen_url_no_comparison(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/unrelated", "whatever")],
        )

        results = _resolve_against_history(
            tmp_path,
            _fred_returning("https://example.com/brand-new", "brand new content"),
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
            [_prior_citation("https://example.com/shared-source", "v1")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/shared-source", "v2")
        )

        assert results[0]["content_changed_since"]["prior_article"] == (
            "some-other-article"
        )

    def test_most_recent_prior_run_wins(self, tmp_path):
        """With several prior sightings of a URL, the comparison is against the
        newest one, not whichever file happened to be scanned last."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "v1")],
        )
        _write_report(
            tmp_path,
            "article-two",
            2,
            "2026-06-01T00:00:00",
            [_prior_citation("https://example.com/data", "v2")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/data", "v3")
        )

        drift = results[0]["content_changed_since"]
        assert drift["prior_article"] == "article-two"
        assert drift["prior_checksum"] == resolver.sha256_checksum("v2")

    def test_missing_history_root_no_crash(self, tmp_path):
        results = _resolve_against_history(
            tmp_path / "does-not-exist", _fred_returning("https://x", "data")
        )

        assert results[0]["resolved"] is True
        assert "content_changed_since" not in results[0]

    def test_no_history_root_skips_history_scan_entirely(self):
        """Callers that don't opt in pay nothing — no history scan happens."""
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=_fred_returning("https://x", "data"),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver."
                "history_analytics.load_reports"
            ) as mock_load,
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        mock_load.assert_not_called()
        assert "content_changed_since" not in results[0]

    def test_known_url_citation_gets_drift_check(self, tmp_path):
        """Drift applies to the known_url path too, not just adapter results."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/page", "old page content")],
        )
        mock_resp = type(
            "R",
            (),
            {"raise_for_status": lambda self: None, "text": "new page content"},
        )()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                history_root=str(tmp_path),
            )

        assert results[0]["verification"] == "checksum"
        assert results[0]["content_changed_since"]["prior_checksum"] == (
            resolver.sha256_checksum("old page content")
        )

    def test_content_mismatch_citation_not_drift_flagged(self, tmp_path):
        """A citation demoted to content_mismatch has left the checksum tier —
        it already carries a stronger signal, so drift doesn't pile on."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/page", "old page content")],
        )
        mock_resp = type(
            "R",
            (),
            {"raise_for_status": lambda self: None, "text": "new page content"},
        )()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
                return_value={
                    "data": {"verdict": "contradicts", "reason": "no"},
                    "model": "mistral-small-latest",
                    "tokens": {},
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
                history_root=str(tmp_path),
            )

        assert results[0]["verification"] == "content_mismatch"
        assert "content_changed_since" not in results[0]


class TestChecksumIndexTierFiltering:
    """Only checksum-tier prior entries are indexed, and only checksum-tier
    resolutions are compared — a pointer-only checksum is taken over whatever
    the adapter called content (often nothing), so comparing it is noise."""

    def test_pointer_only_resolution_is_never_drift_flagged(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/portal", "old blurb")],
        )

        results = _resolve_against_history(
            tmp_path,
            _fred_returning(
                "https://example.com/portal", "new blurb", pointer_only=True
            ),
        )

        assert results[0]["verification"] == "pointer"
        assert "content_changed_since" not in results[0]

    def test_pointer_tier_prior_entry_is_not_indexed(self, tmp_path):
        """A URL last seen pointer-only must not make a later real fetch of
        the same URL look like the page changed."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/portal", "", verification="pointer")],
        )

        assert resolver.build_checksum_index(str(tmp_path)) == {}

    def test_prior_entry_without_verification_field_is_not_indexed(self, tmp_path):
        """Reports predating the confidence tiers can't be placed in a tier,
        so they're skipped rather than compared on a guess."""
        entry = _prior_citation("https://example.com/data", "content")
        del entry["verification"]
        _write_report(tmp_path, "article-one", 1, "2026-01-01T00:00:00", [entry])

        assert resolver.build_checksum_index(str(tmp_path)) == {}

    def test_unresolved_prior_entry_has_no_checksum_to_index(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [
                {
                    "claim": "old claim",
                    "url": "https://example.com/data",
                    "resolved": False,
                    "note": "unfetchable",
                }
            ],
        )

        assert resolver.build_checksum_index(str(tmp_path)) == {}

    def test_checksum_tier_prior_entry_is_indexed(self, tmp_path):
        """Guard against the filters above passing vacuously — a well-formed
        checksum-tier entry must actually land in the index."""
        _write_report(
            tmp_path,
            "article-one",
            7,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "content")],
        )

        index = resolver.build_checksum_index(str(tmp_path))

        assert index["https://example.com/data"] == {
            "checksum": resolver.sha256_checksum("content"),
            "article_slug": "article-one",
            "run_number": 7,
            "generated": "2026-01-01T00:00:00",
        }
