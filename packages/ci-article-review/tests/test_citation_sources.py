"""Tests for citation source adapters — API key guards and basic resolve logic."""

import os

from unittest.mock import patch


class TestEiaResolver:
    def test_returns_not_found_when_no_api_key(self):
        from ci_article_review.adapters.citation.sources import eia

        with patch.dict(os.environ, {}, clear=True):
            # Ensure EIA_API_KEY is absent
            os.environ.pop("EIA_API_KEY", None)
            result = eia.resolve("electricity consumption in Illinois")
        assert result == {"found": False}

    def test_returns_not_found_when_no_energy_keywords(self):
        from ci_article_review.adapters.citation.sources import eia

        with patch.dict(os.environ, {"EIA_API_KEY": "testkey"}):
            result = eia.resolve("population growth in DeKalb County")
        assert result == {"found": False}

    def test_includes_api_key_in_request(self):
        from ci_article_review.adapters.citation.sources import eia
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        with (
            patch.dict(os.environ, {"EIA_API_KEY": "testkey123"}),
            patch(
                "ci_article_review.adapters.citation.sources.eia.requests.get",
                return_value=mock_resp,
            ) as mock_get,
        ):
            eia.resolve("electricity natural gas consumption")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1].get("params") or {}
        assert params.get("api_key") == "testkey123"


class TestFredResolver:
    def test_returns_not_found_when_no_api_key(self):
        from ci_article_review.adapters.citation.sources import fred

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FRED_API_KEY", None)
            result = fred.resolve("unemployment rate in Illinois", api_key=None)
        assert result == {"found": False}


class TestCensusResolver:
    def test_returns_not_found_when_no_api_key(self):
        from ci_article_review.adapters.citation.sources import census

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CENSUS_API_KEY", None)
            result = census.resolve("population in DeKalb County", api_key=None)
        assert result == {"found": False}
