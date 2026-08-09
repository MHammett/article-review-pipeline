"""FERC (Federal Energy Regulatory Commission) resolver.

FERC's eLibrary is a docket/accession-number search with no simple public
claims API, so this is a pointer-only adapter in the FHWA style:
keyword-gate on FERC-relevant terms and point at eLibrary for manual
retrieval.
"""

import logging

from ..topic_match import topic_match

log = logging.getLogger(__name__)

FERC_BASE = "https://elibrary.ferc.gov/eLibrary/search"

FERC_TERMS = [
    "ferc",
    "order 1920",
    "order no. 1920",
    "interconnection order",
    "large-generator interconnection",
    "large generator interconnection",
]


def resolve(claim, api_key=None):
    claim_lower = claim.lower()
    if not topic_match(claim_lower, FERC_TERMS):
        return {"found": False}

    return {
        "found": True,
        "pointer_only": True,
        "url": FERC_BASE,
        "summary": f"FERC eLibrary — manual retrieval required. See {FERC_BASE}",
        "content": f"FERC source pointer for: {claim[:200]}",
    }
