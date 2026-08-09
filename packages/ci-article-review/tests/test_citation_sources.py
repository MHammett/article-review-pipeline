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


class TestCrossrefResolver:
    def test_returns_not_found_for_unrelated_claim(self):
        from ci_article_review.adapters.citation.sources import crossref

        result = crossref.resolve("The county board approved the tax levy last week.")
        assert result == {"found": False}

    def test_resolves_literal_doi(self):
        from ci_article_review.adapters.citation.sources import crossref
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "title": ["Data Center Water Use in the American Southwest"],
                "author": [{"given": "J.", "family": "Smith"}],
                "container-title": ["Environmental Research Letters"],
                "publisher": "IOP Publishing",
                "published-print": {"date-parts": [[2025, 3, 1]]},
                "DOI": "10.1029/2025av002140",
            }
        }
        with patch(
            "ci_article_review.adapters.citation.sources.crossref.requests.get",
            return_value=mock_resp,
        ) as mock_get:
            result = crossref.resolve(
                "A study found (doi.org/10.1029/2025AV002140) that water use rose sharply."
            )
        assert result["found"] is True
        assert result["url"] == "https://doi.org/10.1029/2025av002140"
        assert "Smith" in result["summary"]
        # DOI lookup hits works/{doi} directly, not the search endpoint.
        assert "works/10.1029/2025av002140" in mock_get.call_args[0][0]

    def test_bibliographic_search_match_above_confidence_threshold(self):
        from ci_article_review.adapters.citation.sources import crossref
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "items": [
                    {
                        "title": ["Data Center Water Use in the American Southwest"],
                        "author": [{"given": "J.", "family": "Smith"}],
                        "container-title": ["Environmental Research Letters"],
                        "DOI": "10.1029/2025av002140",
                    }
                ]
            }
        }
        claim = (
            'A peer-reviewed study titled "Data Center Water Use in the American '
            'Southwest" found consumption rose sharply.'
        )
        with patch(
            "ci_article_review.adapters.citation.sources.crossref.requests.get",
            return_value=mock_resp,
        ):
            result = crossref.resolve(claim)
        assert result["found"] is True
        assert result["url"] == "https://doi.org/10.1029/2025av002140"

    def test_bibliographic_search_below_confidence_threshold_not_found(self):
        from ci_article_review.adapters.citation.sources import crossref
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "items": [
                    {
                        "title": ["Completely Unrelated Title About Mollusk Genetics"],
                        "DOI": "10.9999/xyz",
                    }
                ]
            }
        }
        claim = "A recent study published in a journal found that grid reliability improved."
        with patch(
            "ci_article_review.adapters.citation.sources.crossref.requests.get",
            return_value=mock_resp,
        ):
            result = crossref.resolve(claim)
        assert result == {"found": False}


class TestEpaResolver:
    def test_returns_not_found_for_unrelated_claim(self):
        from ci_article_review.adapters.citation.sources import epa

        result = epa.resolve("The county board approved the tax levy last week.")
        assert result == {"found": False}

    def test_resolves_pfas_claim_as_pointer(self):
        from ci_article_review.adapters.citation.sources import epa

        result = epa.resolve("New PFAS limits were set for community water systems.")
        assert result["found"] is True
        assert result["pointer_only"] is True
        assert "pfas" in result["url"].lower()

    def test_resolves_air_quality_claim_as_pointer(self):
        from ci_article_review.adapters.citation.sources import epa

        result = epa.resolve("The facility exceeded NAAQS thresholds for PM2.5.")
        assert result["found"] is True
        assert result["pointer_only"] is True


class TestPjmResolver:
    def test_returns_not_found_for_unrelated_claim(self):
        from ci_article_review.adapters.citation.sources import pjm

        result = pjm.resolve("The county board approved the tax levy last week.")
        assert result == {"found": False}

    def test_resolves_capacity_auction_claim(self):
        from ci_article_review.adapters.citation.sources import pjm

        result = pjm.resolve("PJM's latest capacity auction cleared at a record price.")
        assert result["found"] is True
        assert result["pointer_only"] is True


class TestIccResolver:
    def test_returns_not_found_for_unrelated_claim(self):
        from ci_article_review.adapters.citation.sources import icc

        result = icc.resolve("The county board approved the tax levy last week.")
        assert result == {"found": False}

    def test_extracts_docket_number(self):
        from ci_article_review.adapters.citation.sources import icc

        result = icc.resolve("ICC Docket 24-0181 addressed ComEd's rate case.")
        assert result["found"] is True
        assert result["pointer_only"] is True
        assert "24-0181" in result["url"]


class TestFercResolver:
    def test_returns_not_found_for_unrelated_claim(self):
        from ci_article_review.adapters.citation.sources import ferc

        result = ferc.resolve("The county board approved the tax levy last week.")
        assert result == {"found": False}

    def test_resolves_order_1920_claim(self):
        from ci_article_review.adapters.citation.sources import ferc

        result = ferc.resolve("FERC's Order 1920 changes transmission planning rules.")
        assert result["found"] is True
        assert result["pointer_only"] is True


class TestIlgaResolver:
    def test_returns_not_found_for_unrelated_claim(self):
        from ci_article_review.adapters.citation.sources import ilga

        result = ilga.resolve("The county board approved the tax levy last week.")
        assert result == {"found": False}

    def test_extracts_public_act_number(self):
        from ci_article_review.adapters.citation.sources import ilga

        result = ilga.resolve("Public Act 103-580 amended the Illinois utility code.")
        assert result["found"] is True
        assert result["pointer_only"] is True
        assert "103580" in result["url"]
