"""EIA (Energy Information Administration) data resolver."""
import requests
import logging

EIA_BASE = "https://api.eia.gov/v2"
log = logging.getLogger(__name__)


def resolve(claim, api_key=None):
    """
    Minimal resolver: searches EIA dataset catalog for terms from the claim.
    Full implementation requires an EIA API key set in environment.
    """
    keywords = _extract_keywords(claim)
    if not keywords:
        return {"found": False}

    try:
        url = f"{EIA_BASE}/seriesid/{'+'.join(keywords[:2])}"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "found": True,
                "url": url,
                "summary": str(data)[:500],
                "content": str(data),
            }
    except Exception as e:
        log.debug(f"EIA resolve failed: {e}")

    return {"found": False}


def _extract_keywords(claim):
    energy_terms = ["electricity", "natural gas", "petroleum", "coal", "renewable",
                    "kwh", "mwh", "btu", "barrel", "mcf"]
    return [t for t in energy_terms if t.lower() in claim.lower()]
