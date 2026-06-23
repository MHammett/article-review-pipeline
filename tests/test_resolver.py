"""Tests for adapters.citation.resolver — parallel resolution, ordering, pointer flag."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch

from adapters.citation import resolver


_SOURCES = [{"name": "FRED", "adapter": "fred"}]


def _no_wayback(url, timeout=10):
    return {"archived": None}


class TestResolveCitations:
    def test_empty_claims(self):
        assert resolver.resolve_citations([], _SOURCES) == []

    def test_preserves_claim_order(self):
        claims = ["claim zero", "claim one", "claim two"]

        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": f"https://x/{claim.split()[-1]}", "content": claim}

        with patch("adapters.citation.resolver.wayback.check", side_effect=_no_wayback), \
             patch("adapters.citation.sources.fred.resolve", side_effect=fake_resolve):
            results = resolver.resolve_citations(claims, _SOURCES)

        assert [r["claim"] for r in results] == claims

    def test_unresolved_claim_marked(self):
        def fake_resolve(claim, api_key=None):
            return {"found": False}

        with patch("adapters.citation.sources.fred.resolve", side_effect=fake_resolve):
            results = resolver.resolve_citations(["unknown claim"], _SOURCES)

        assert results[0]["resolved"] is False
        assert "note" in results[0]

    def test_checksum_verification_label(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with patch("adapters.citation.resolver.wayback.check", side_effect=_no_wayback), \
             patch("adapters.citation.sources.fred.resolve", side_effect=fake_resolve):
            results = resolver.resolve_citations(["c"], _SOURCES)

        assert results[0]["verification"] == "checksum"

    def test_pointer_only_label(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "pointer_only": True, "url": "https://x", "content": "ptr"}

        with patch("adapters.citation.resolver.wayback.check", side_effect=_no_wayback), \
             patch("adapters.citation.sources.fred.resolve", side_effect=fake_resolve):
            results = resolver.resolve_citations(["c"], _SOURCES)

        assert results[0]["verification"] == "pointer"

    def test_missing_source_name_does_not_crash(self):
        sources = [{"adapter": "fred"}]  # no "name" key

        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with patch("adapters.citation.resolver.wayback.check", side_effect=_no_wayback), \
             patch("adapters.citation.sources.fred.resolve", side_effect=fake_resolve):
            results = resolver.resolve_citations(["c"], sources)

        # Falls back to adapter name instead of raising KeyError
        assert results[0]["source_name"] == "fred"

    def test_adapter_exception_is_isolated(self):
        def boom(claim, api_key=None):
            raise RuntimeError("adapter exploded")

        with patch("adapters.citation.sources.fred.resolve", side_effect=boom):
            results = resolver.resolve_citations(["c"], _SOURCES)

        # Exception is caught; claim reported unresolved rather than crashing the run
        assert results[0]["resolved"] is False
