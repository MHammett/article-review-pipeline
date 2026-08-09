"""Illinois Commerce Commission resolver.

ICC's e-Docket system is a docket-number search, not a claims API, so this
is a pointer-only adapter in the FHWA style: keyword-gate on ICC-relevant
terms and point at the e-Docket search page for manual retrieval.
"""

import logging
import re

log = logging.getLogger(__name__)

ICC_BASE = "https://icc.illinois.gov/docket/dockets.aspx"

ICC_TERMS = [
    "illinois commerce commission",
    " icc ",
    "icc docket",
    "rate case",
    "multi-year plan",
    "mypp",
    "tariff filing",
    "comed rate",
]

_DOCKET_RE = re.compile(r"\b(\d{2}-\d{4})\b")


def resolve(claim, api_key=None):
    claim_lower = f" {claim.lower()} "
    if not any(t in claim_lower for t in ICC_TERMS):
        return {"found": False}

    docket_match = _DOCKET_RE.search(claim)
    if docket_match:
        docket = docket_match.group(1)
        url = f"{ICC_BASE}?docketNo={docket}"
        summary = f"ICC Docket {docket} — manual retrieval required. See {url}"
    else:
        url = ICC_BASE
        summary = f"Illinois Commerce Commission e-Docket search — manual retrieval required. See {url}"

    return {
        "found": True,
        "pointer_only": True,
        "url": url,
        "summary": summary,
        "content": f"ICC source pointer for: {claim[:200]}",
    }
