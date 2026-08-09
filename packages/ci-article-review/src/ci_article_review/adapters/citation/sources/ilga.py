"""Illinois General Assembly resolver.

ilga.gov's bill/public act search has no simple public claims API, so this
is a pointer-only adapter in the FHWA style: keyword-gate on ILGA-relevant
terms, extract a public act or bill number if present, and point at the
right ilga.gov search for manual retrieval.
"""

import logging
import re

from ..topic_match import topic_match

log = logging.getLogger(__name__)

ILGA_BASE = "https://www.ilga.gov"

ILGA_TERMS = [
    "illinois general assembly",
    "ilga",
    "public act",
    "illinois senate bill",
    "illinois house bill",
]

_PUBLIC_ACT_RE = re.compile(r"\bpublic act (\d{2,3}-\d{3,4})\b", re.IGNORECASE)
_BILL_RE = re.compile(r"\b([SH]B)\s?0*(\d{1,5})\b", re.IGNORECASE)


def resolve(claim, api_key=None):
    claim_lower = claim.lower()
    if not topic_match(claim_lower, ILGA_TERMS):
        return {"found": False}

    pa_match = _PUBLIC_ACT_RE.search(claim)
    if pa_match:
        pa_number = pa_match.group(1)
        url = f"{ILGA_BASE}/legislation/publicacts/{pa_number.replace('-', '')}.html"
        summary = (
            f"Illinois Public Act {pa_number} — manual retrieval required. See {url}"
        )
        return {
            "found": True,
            "pointer_only": True,
            "url": url,
            "summary": summary,
            "content": f"ILGA source pointer (Public Act {pa_number}) for: {claim[:200]}",
        }

    bill_match = _BILL_RE.search(claim)
    if bill_match:
        chamber, number = bill_match.group(1).upper(), bill_match.group(2)
        url = f"{ILGA_BASE}/legislation/BillStatus.asp?DocNum={number}&DocTypeID={chamber}"
        summary = f"Illinois {chamber} {number} — manual retrieval required. See {url}"
        return {
            "found": True,
            "pointer_only": True,
            "url": url,
            "summary": summary,
            "content": f"ILGA source pointer ({chamber} {number}) for: {claim[:200]}",
        }

    url = f"{ILGA_BASE}/legislation/"
    return {
        "found": True,
        "pointer_only": True,
        "url": url,
        "summary": f"Illinois General Assembly legislation search — manual retrieval required. See {url}",
        "content": f"ILGA source pointer for: {claim[:200]}",
    }
