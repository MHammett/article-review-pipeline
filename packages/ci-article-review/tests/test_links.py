"""Tests for analysis.links — URL extraction, SSRF guard, parallel validation."""
import sys, os

from unittest.mock import patch, MagicMock

from ci_article_review.analysis import links


class TestExtractUrls:
    def test_strips_trailing_period(self):
        assert links.extract_urls("See https://example.com.") == ["https://example.com"]

    def test_strips_trailing_comma(self):
        assert links.extract_urls("https://example.com, and more") == ["https://example.com"]

    def test_preserves_query_string(self):
        urls = links.extract_urls("Go to https://example.com/p?q=1&r=2 now")
        assert urls == ["https://example.com/p?q=1&r=2"]

    def test_dedupes(self):
        text = "https://a.com and https://a.com again"
        assert links.extract_urls(text) == ["https://a.com"]

    def test_no_urls(self):
        assert links.extract_urls("no links here") == []


class TestSsrfGuard:
    def test_blocks_cloud_metadata(self):
        assert links._is_public_host("http://169.254.169.254/latest/meta-data/") is False

    def test_blocks_localhost(self):
        assert links._is_public_host("http://localhost:6379/") is False

    def test_blocks_loopback_ip(self):
        assert links._is_public_host("http://127.0.0.1/admin") is False

    def test_blocks_private_ranges(self):
        for url in ("http://192.168.1.1/", "http://10.0.0.5/", "http://172.16.0.1/"):
            assert links._is_public_host(url) is False, url

    def test_allows_public_host(self):
        assert links._is_public_host("https://www.fhwa.dot.gov/") is True

    def test_no_hostname_rejected(self):
        assert links._is_public_host("not-a-url") is False

    def test_check_http_skips_internal(self):
        result = links._check_http("http://169.254.169.254/")
        assert result["ok"] is False
        assert "SSRF guard" in result["error"]


class TestValidateLinks:
    def test_empty_text_returns_empty(self):
        assert links.validate_links("", check_wayback=False) == []

    def test_preserves_url_order(self):
        text = "First https://zzz.example.com then https://aaa.example.com"
        # Mock the per-URL worker so no network happens
        def fake_check(url, timeout):
            return {"status_code": 200, "ok": True, "redirected_to": None}
        with patch("ci_article_review.analysis.links._check_http", side_effect=fake_check):
            results = links.validate_links(text, check_wayback=False)
        assert [r["url"] for r in results] == ["https://zzz.example.com", "https://aaa.example.com"]

    def test_internal_url_not_sent_to_wayback(self):
        text = "Internal http://localhost:8080/x"
        with patch("ci_article_review.analysis.links.wayback_check") as mock_wb:
            results = links.validate_links(text, check_wayback=True)
        # SSRF-guarded host should never hit Wayback
        mock_wb.assert_not_called()
        assert results[0]["ok"] is False

    def test_wayback_stale_days_threaded_through(self):
        # The wayback_stale_days arg must reach wayback_check as stale_days.
        text = "See https://example.com"
        def fake_http(url, timeout):
            return {"status_code": 200, "ok": True, "redirected_to": None}
        with patch("ci_article_review.analysis.links._check_http", side_effect=fake_http), \
             patch("ci_article_review.analysis.links.wayback_check", return_value={"archived": True}) as mock_wb:
            links.validate_links(text, check_wayback=True, wayback_stale_days=90)
        _, kwargs = mock_wb.call_args
        assert kwargs.get("stale_days") == 90
