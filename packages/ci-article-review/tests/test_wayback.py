"""Tests for adapters.citation.wayback (all HTTP mocked)."""

from unittest.mock import patch, MagicMock

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
        summary = wayback.format_summary(
            {"url": "https://x.com", "archived": None, "error": "timeout"}
        )
        assert "failed" in summary.lower()

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
