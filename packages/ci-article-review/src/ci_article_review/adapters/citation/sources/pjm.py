"""PJM Interconnection resolver.

PJM's machine-readable data (Data Miner 2) requires a subscription key and is
scoped to specific report names, not free-text claims, so this is a
pointer-only adapter in the FHWA style: keyword-gate on PJM-relevant terms
and point at the right public page for manual retrieval.
"""

import logging

log = logging.getLogger(__name__)

PJM_TERMS = [
    "pjm",
    "capacity auction",
    "teac",
    "elcc",
    "irm",
    "reliability pricing model",
    "rpm",
    "interconnection queue",
    "generation interconnection",
]


def resolve(claim, api_key=None):
    claim_lower = claim.lower()
    if not any(t in claim_lower for t in PJM_TERMS):
        return {"found": False}

    if "capacity auction" in claim_lower or "rpm" in claim_lower or "reliability pricing model" in claim_lower:
        url = "https://www.pjm.com/markets-and-operations/rpm"
        program = "Capacity (RPM) Auction Results"
    elif "interconnection queue" in claim_lower or "generation interconnection" in claim_lower or "teac" in claim_lower:
        url = "https://www.pjm.com/planning/service-requests"
        program = "Generation Interconnection Queue / TEAC"
    else:
        url = "https://www.pjm.com/markets-and-operations"
        program = "Markets & Operations"

    return {
        "found": True,
        "pointer_only": True,
        "url": url,
        "summary": f"PJM {program} — manual retrieval required. See {url}",
        "content": f"PJM source pointer ({program}) for: {claim[:200]}",
    }
