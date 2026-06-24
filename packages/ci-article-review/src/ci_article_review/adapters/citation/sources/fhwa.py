"""FHWA (Federal Highway Administration) Highway Statistics resolver."""

import logging

FHWA_BASE = "https://www.fhwa.dot.gov/policyinformation/statistics"
log = logging.getLogger(__name__)


def resolve(claim, api_key=None):
    """
    FHWA does not have a machine-readable API; this adapter returns a
    pointer to the relevant Highway Statistics publication for manual retrieval.
    """
    claim_lower = claim.lower()
    highway_terms = [
        "vehicle miles",
        "vmt",
        "lane miles",
        "highway",
        "road",
        "bridge",
        "traffic",
        "fuel tax",
        "motor fuel",
    ]

    if not any(t in claim_lower for t in highway_terms):
        return {"found": False}

    year_match = _extract_year(claim)
    year = year_match or "latest"
    url = f"{FHWA_BASE}/{year}/" if year != "latest" else f"{FHWA_BASE}/"

    return {
        "found": True,
        "pointer_only": True,  # not a verified data fetch — points at a publication for manual retrieval
        "url": url,
        "summary": f"FHWA Highway Statistics ({year}) — manual retrieval required. See {url}",
        "content": f"FHWA source pointer for: {claim[:200]}",
    }


def _extract_year(text):
    import re

    match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return match.group(0) if match else None
