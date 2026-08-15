"""Tests for adapters.citation.wayback (all HTTP mocked)."""

import itertools
import threading
import time
from unittest.mock import patch, MagicMock

import requests

from ci_article_review.adapters.citation import wayback


def _mock_response(payload):
    m = MagicMock()
    m.json.return_value = payload
    m.raise_for_status.return_value = None
    return m


class TestWaybackCheck:
    def test_archived_url(self):
        payload = {
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "url": "https://web.archive.org/web/20250101000000/https://example.com",
                    "timestamp": "20250101000000",
                }
            }
        }
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get",
            return_value=_mock_response(payload),
        ):
            result = wayback.check("https://example.com")
        assert result["archived"] is True
        assert result["snapshot_url"].startswith("https://web.archive.org")
        assert result["snapshot_age_days"] is not None
        assert isinstance(result["snapshot_stale"], bool)

    def test_not_archived(self):
        payload = {"archived_snapshots": {}}
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get",
            return_value=_mock_response(payload),
        ):
            result = wayback.check("https://example.com")
        assert result["archived"] is False
        assert "snapshot_url" not in result

    def test_network_error_returns_none_archived(self):
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get",
            side_effect=Exception("timeout"),
        ):
            result = wayback.check("https://example.com")
        assert result["archived"] is None
        assert "error" in result

    def test_stale_snapshot_flagged(self):
        # Snapshot from 2020 should be stale (>180 days)
        payload = {
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "url": "https://web.archive.org/web/20200101000000/https://old.com",
                    "timestamp": "20200101000000",
                }
            }
        }
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get",
            return_value=_mock_response(payload),
        ):
            result = wayback.check("https://old.com")
        assert result["snapshot_stale"] is True

    def test_fresh_snapshot_not_stale(self):
        from datetime import datetime, timezone, timedelta

        recent_ts = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
            "%Y%m%d"
        ) + "000000"
        payload = {
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "url": f"https://web.archive.org/web/{recent_ts}/https://fresh.com",
                    "timestamp": recent_ts,
                }
            }
        }
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get",
            return_value=_mock_response(payload),
        ):
            result = wayback.check("https://fresh.com")
        assert result["snapshot_stale"] is False

    def test_custom_stale_days_override(self):
        from datetime import datetime, timezone, timedelta

        # Snapshot from 100 days ago: NOT stale at default 180, but stale at a custom 90.
        ts = (datetime.now(timezone.utc) - timedelta(days=100)).strftime(
            "%Y%m%d"
        ) + "000000"
        payload = {
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "url": f"https://web.archive.org/web/{ts}/https://x.com",
                    "timestamp": ts,
                }
            }
        }
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get",
            return_value=_mock_response(payload),
        ):
            default_result = wayback.check("https://x.com")
            strict_result = wayback.check("https://x.com", stale_days=90)
        assert default_result["snapshot_stale"] is False
        assert strict_result["snapshot_stale"] is True

    def test_archive_url_detected_without_api_call(self):
        # A web.archive.org link is recognized from its embedded timestamp — no
        # archive-of-an-archive lookup. requests.get must not be called.
        from datetime import datetime, timezone, timedelta

        recent = (datetime.now(timezone.utc) - timedelta(days=20)).strftime(
            "%Y%m%d"
        ) + "120000"
        url = f"https://web.archive.org/web/{recent}/https://example.com/article"
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get"
        ) as mock_get:
            result = wayback.check(url)
        mock_get.assert_not_called()
        assert result["is_archive_url"] is True
        assert result["archived"] is True
        assert result["snapshot_age_days"] == 20
        assert result["snapshot_stale"] is False

    def test_archive_url_stale_flagged(self):
        url = "https://web.archive.org/web/20200101000000/https://old.example.com"
        result = wayback.check(url)
        assert result["is_archive_url"] is True
        assert result["snapshot_stale"] is True

    def test_archive_url_respects_custom_stale_days(self):
        from datetime import datetime, timezone, timedelta

        ts = (datetime.now(timezone.utc) - timedelta(days=100)).strftime(
            "%Y%m%d"
        ) + "000000"
        url = f"https://web.archive.org/web/{ts}/https://example.com"
        assert wayback.check(url)["snapshot_stale"] is False  # default 180
        assert wayback.check(url, stale_days=90)["snapshot_stale"] is True

    def test_non_archive_url_still_queries_api(self):
        # Regression: ordinary URLs must still hit the availability API.
        payload = {"archived_snapshots": {}}
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get",
            return_value=_mock_response(payload),
        ) as mock_get:
            wayback.check("https://example.com/normal")
        mock_get.assert_called_once()

    def test_format_summary_not_archived(self):
        summary = wayback.format_summary({"url": "https://x.com", "archived": False})
        assert "Not archived" in summary

    def test_format_summary_network_error(self):
        """A null must not read as "not archived" — it means we never asked.

        This asserted only that the word "failed" appeared, which the old
        exception-led wording satisfied while still leaving a reader to infer
        what a failed check implied about the page. It now has to say so.
        """
        summary = wayback.format_summary(
            {"url": "https://x.com", "archived": None, "error": "timeout"}
        )
        assert "NOT CHECKED" in summary
        assert "timeout" in summary
        assert "says nothing about whether the page is archived" in summary
        assert "Not archived" not in summary

    def test_format_summary_archived(self):
        wb = {
            "archived": True,
            "snapshot_age_days": 30,
            "snapshot_stale": False,
            "snapshot_url": "https://web.archive.org/...",
        }
        summary = wayback.format_summary(wb)
        assert "30d" in summary
        assert "[STALE]" not in summary


class TestWaybackSubmit:
    def test_unauthenticated_submit_success(self):
        resp = _mock_response({})
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get",
            return_value=resp,
        ) as mock_get:
            result = wayback.submit("https://example.com")
        assert result["submitted"] is True
        assert result["job_id"] is None
        # No credentials given — falls back to the unauthenticated trigger endpoint.
        mock_get.assert_called_once()
        assert (
            "https://web.archive.org/save/https://example.com"
            in mock_get.call_args[0][0]
        )

    def test_authenticated_submit_success(self):
        resp = _mock_response({"job_id": "spn2-abc123"})
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.post",
            return_value=resp,
        ) as mock_post:
            result = wayback.submit(
                "https://example.com",
                access_key="AK123",
                secret_key="SK456",
            )
        assert result["submitted"] is True
        assert result["job_id"] == "spn2-abc123"
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "LOW AK123:SK456"

    def test_submit_failure_does_not_raise(self):
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.get",
            side_effect=Exception("rate limited"),
        ):
            result = wayback.submit("https://example.com")
        assert result["submitted"] is False
        assert "rate limited" in result["error"]

    def test_submit_authenticated_failure_redacts_secret(self):
        with patch(
            "ci_article_review.adapters.citation.wayback.requests.post",
            side_effect=Exception("auth failed for SK456"),
        ):
            result = wayback.submit(
                "https://example.com",
                access_key="AK123",
                secret_key="SK456",
            )
        assert result["submitted"] is False
        assert "SK456" not in result["error"]


class TestFallbackScoping:
    """Which failures an archive snapshot may stand in for — the policy both
    analysis/links.py and the citation resolver read from."""

    def test_origin_refusals_qualify(self):
        assert wayback.fallback_reason_for_status(401) == "auth_required"
        assert wayback.fallback_reason_for_status(403) == "blocked"
        assert wayback.fallback_reason_for_status(429) == "rate_limited"

    def test_gone_and_origin_errors_do_not_qualify(self):
        # 404/410: the resource is genuinely gone and that must surface.
        # 5xx: the origin's own failure, not a refusal aimed at us.
        for status in (200, 404, 410, 500, 502, 503, None):
            assert wayback.fallback_reason_for_status(status) is None

    def test_unreachable_origins_qualify(self):
        assert (
            wayback.fallback_reason_for_exception(requests.exceptions.Timeout())
            == "timeout"
        )
        assert (
            wayback.fallback_reason_for_exception(requests.exceptions.ReadTimeout())
            == "timeout"
        )
        # ConnectTimeout subclasses both Timeout and ConnectionError; the more
        # specific "timeout" wins.
        assert (
            wayback.fallback_reason_for_exception(requests.exceptions.ConnectTimeout())
            == "timeout"
        )
        assert (
            wayback.fallback_reason_for_exception(
                requests.exceptions.ConnectionError("NameResolutionError")
            )
            == "unreachable"
        )

    def test_http_error_dispatches_on_status(self):
        resp = MagicMock(status_code=403)
        exc = requests.exceptions.HTTPError("403", response=resp)
        assert wayback.fallback_reason_for_exception(exc) == "blocked"

        resp404 = MagicMock(status_code=404)
        gone = requests.exceptions.HTTPError("404", response=resp404)
        assert wayback.fallback_reason_for_exception(gone) is None

    def test_http_error_without_response_does_not_qualify(self):
        exc = requests.exceptions.HTTPError("no response attached")
        assert wayback.fallback_reason_for_exception(exc) is None

    def test_unrelated_exception_does_not_qualify(self):
        assert wayback.fallback_reason_for_exception(ValueError("nope")) is None

    def test_every_reason_has_a_label(self):
        reasons = set(wayback._FALLBACK_STATUSES.values()) | {"timeout", "unreachable"}
        assert reasons <= set(wayback.FALLBACK_REASON_LABELS)


def _rate_limited_response(retry_after="0"):
    """A 429. ``Retry-After: 0`` keeps the shared clock from actually sleeping,
    which is what makes these tests fast; the one test that cares about the
    clock moving passes a real value."""
    m = MagicMock()
    m.status_code = 429
    m.url = "https://archive.org/wayback/available"
    m.headers = {"Retry-After": retry_after}
    return m


def _ok_response():
    m = MagicMock()
    m.status_code = 200
    m.headers = {}
    m.json.return_value = {"archived_snapshots": {}}
    m.raise_for_status.return_value = None
    return m


class TestRateLimitGuard:
    """The guard is process-wide state driven from a resolver thread pool.

    Every test here resets that state first, because it outlives a single test
    exactly the way it outlives a single run — which is the bug the reset in
    ``run_draft_pipeline`` exists to prevent.
    """

    def setup_method(self):
        wayback.reset_rate_limit_state()

    def teardown_method(self):
        wayback.reset_rate_limit_state()

    def test_breaker_trips_when_most_lookups_are_refused(self):
        """A success must not erase other threads' refusals.

        Eight workers against an endpoint refusing four of every five: the run
        is plainly being throttled and the breaker has to trip. Resetting a
        shared counter on success made this run forever, because some thread
        was always succeeding and zeroing the count.
        """
        import concurrent.futures

        counter = itertools.count()
        lock = threading.Lock()

        def flaky_get(*args, **kwargs):
            with lock:
                n = next(counter)
            return _ok_response() if n % 5 == 4 else _rate_limited_response()

        with (
            patch(
                "ci_article_review.adapters.citation.wayback.requests.get",
                side_effect=flaky_get,
            ),
            patch.object(wayback, "_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(wayback, "_BACKOFF_BASE_SECONDS", 0.0),
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(
                    pool.map(
                        lambda i: wayback.check(f"https://example.com/{i}"),
                        range(24),
                    )
                )

        assert wayback.rate_limited_out() is True

    def test_a_refused_lookup_counts_once_not_once_per_attempt(self):
        """``_CIRCUIT_TRIP_AFTER`` counts lookups, so it must mean lookups.

        Counting each retry inside a lookup tripped the breaker after two URLs
        while the constant said five.
        """
        with (
            patch(
                "ci_article_review.adapters.citation.wayback.requests.get",
                return_value=_rate_limited_response(),
            ),
            patch.object(wayback, "_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(wayback, "_BACKOFF_BASE_SECONDS", 0.0),
        ):
            wayback.check("https://example.com/one")
            assert wayback._rate_limited_lookups == 1
            assert wayback.rate_limited_out() is False

            for i in range(wayback._CIRCUIT_TRIP_AFTER - 1):
                wayback.check(f"https://example.com/{i}")

        assert wayback._rate_limited_lookups == wayback._CIRCUIT_TRIP_AFTER
        assert wayback.rate_limited_out() is True

    def test_a_429_backs_off_every_thread_not_just_the_one_that_hit_it(self):
        """The backoff has to land on shared state, or it is not a backoff.

        Sleeping inside the failing thread left the other workers calling at
        full pace, so the process never actually slowed down.
        """
        with (
            patch(
                "ci_article_review.adapters.citation.wayback.requests.get",
                return_value=_rate_limited_response(retry_after="30"),
            ),
            patch.object(wayback, "_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(wayback, "_BACKOFF_BASE_SECONDS", 0.0),
        ):
            before = time.monotonic()
            # One attempt is enough to prove the clock moved; the retries would
            # each sleep out the 30s backoff this test is asserting exists.
            with patch.object(wayback, "_MAX_ATTEMPTS", 1):
                wayback.check("https://example.com/blocked")

        # Retry-After: 30 pushes the shared clock into the future for everyone,
        # not just for the thread that collected the 429.
        assert wayback._blocked_until >= before + 30.0

    def test_reset_clears_a_tripped_breaker(self):
        """Without this, run N+1 in the same process skips every lookup."""
        with (
            patch(
                "ci_article_review.adapters.citation.wayback.requests.get",
                return_value=_rate_limited_response(),
            ),
            patch.object(wayback, "_MIN_INTERVAL_SECONDS", 0.0),
            patch.object(wayback, "_BACKOFF_BASE_SECONDS", 0.0),
        ):
            for i in range(wayback._CIRCUIT_TRIP_AFTER):
                wayback.check(f"https://example.com/{i}")

        assert wayback.rate_limited_out() is True
        wayback.reset_rate_limit_state()
        assert wayback.rate_limited_out() is False
        assert wayback._blocked_until == 0.0

    def test_a_success_still_returns_normally(self):
        """The budget must not break the ordinary path."""
        with (
            patch(
                "ci_article_review.adapters.citation.wayback.requests.get",
                return_value=_ok_response(),
            ),
            patch.object(wayback, "_MIN_INTERVAL_SECONDS", 0.0),
        ):
            result = wayback.check("https://example.com")

        assert result["archived"] is False
        assert wayback.rate_limited_out() is False


class TestNonJsonAvailabilityResponse:
    def setup_method(self):
        wayback.reset_rate_limit_state()

    def teardown_method(self):
        wayback.reset_rate_limit_state()

    def test_non_json_200_is_not_reported_as_a_parse_error(self):
        """Say what arrived, not what the JSON decoder thought of it.

        The raw decoder message sends the reader after a parser bug that isn't
        there — the misdirection reported upstream as akamhy/waybackpy#200.
        """
        m = MagicMock()
        m.status_code = 200
        m.headers = {}
        m.content = b"<html><body>archive.org is temporarily unavailable</body></html>"
        m.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        m.raise_for_status.return_value = None

        with (
            patch(
                "ci_article_review.adapters.citation.wayback.requests.get",
                return_value=m,
            ),
            patch.object(wayback, "_MIN_INTERVAL_SECONDS", 0.0),
        ):
            result = wayback.check("https://example.com")

        assert result["archived"] is None
        assert "non-JSON 200 response" in result["error"]
        assert "Expecting value" not in result["error"]
