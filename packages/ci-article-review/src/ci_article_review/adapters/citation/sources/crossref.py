"""Crossref DOI / bibliographic metadata resolver.

No API key required. Crossref asks callers to identify themselves with a
"polite pool" User-Agent (a mailto contact) for better rate limits — set
CROSSREF_MAILTO if you have one; a generic default is used otherwise.
"""

import difflib
import json
import logging
import os
import re

import requests

CROSSREF_BASE = "https://api.crossref.org"
log = logging.getLogger(__name__)

# Matches a bare DOI (e.g. "10.1029/2025AV002140") whether written on its own
# or inside a doi.org URL. DOI syntax is loosely "10.NNNN/suffix" where the
# suffix is any run of non-whitespace; trailing sentence punctuation is
# stripped separately since it's not part of the identifier.
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)

# A search match is only trusted if the claim looks like it's citing academic
# research in the first place — otherwise the bibliographic search endpoint
# would be run against every claim and would eventually false-match something.
_ACADEMIC_TERMS = [
    "study",
    "studies",
    "researcher",
    "research",
    "journal",
    "paper",
    "published in",
    "peer-reviewed",
    "peer reviewed",
    "findings",
    "et al",
    "doi.org",
    "doi:",
]

# Minimum title-similarity ratio (difflib SequenceMatcher) required to treat a
# bibliographic search result as a genuine match rather than a coincidence.
_SEARCH_CONFIDENCE_THRESHOLD = 0.55

_QUOTED_RE = re.compile(r"[\"“]([^\"”]{15,200})[\"”]")


def resolve(claim, api_key=None):
    mailto = os.getenv("CROSSREF_MAILTO", "citations@ci-article-review.local")
    headers = {"User-Agent": f"ci-article-review/1.0 (mailto:{mailto})"}

    doi = _extract_doi(claim)
    if doi:
        result = _lookup_doi(doi, headers)
        if result:
            return result

    if not _looks_academic(claim):
        return {"found": False}

    result = _search_bibliographic(claim, headers)
    if result:
        return result

    return {"found": False}


def _extract_doi(claim):
    match = _DOI_RE.search(claim)
    if not match:
        return None
    # DOIs are case-insensitive by spec; Crossref itself normalizes to
    # lowercase, so match that convention here.
    return match.group(0).rstrip(".,;:)]”\"'").lower()


def _lookup_doi(doi, headers):
    try:
        resp = requests.get(f"{CROSSREF_BASE}/works/{doi}", headers=headers, timeout=15)
        if resp.status_code != 200:
            log.debug("Crossref DOI lookup HTTP %s for %s", resp.status_code, doi)
            return None
        message = resp.json().get("message", {})
        return _build_result(message, f"https://doi.org/{doi}")
    except Exception as e:
        log.debug(f"Crossref DOI lookup failed: {e}")
        return None


def _search_bibliographic(claim, headers):
    query = _extract_quoted_title(claim) or claim
    try:
        resp = requests.get(
            f"{CROSSREF_BASE}/works",
            params={"query.bibliographic": query, "rows": 1},
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            log.debug("Crossref search HTTP %s", resp.status_code)
            return None
        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return None
        item = items[0]
        title = " ".join(item.get("title") or [])
        if not title:
            return None
        confidence = difflib.SequenceMatcher(None, query.lower(), title.lower()).ratio()
        if confidence < _SEARCH_CONFIDENCE_THRESHOLD:
            log.debug(
                "Crossref search match below confidence threshold (%.2f): %s",
                confidence,
                title,
            )
            return None
        doi = item.get("DOI")
        url = f"https://doi.org/{doi}" if doi else item.get("URL")
        return _build_result(item, url)
    except Exception as e:
        log.debug(f"Crossref bibliographic search failed: {e}")
        return None


def _build_result(message, url):
    if not message:
        return None
    title = " ".join(message.get("title") or []) or "(untitled)"
    authors = message.get("author") or []
    author_names = ", ".join(
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in authors
        if a.get("family")
    )
    container = " ".join(message.get("container-title") or [])
    publisher = message.get("publisher", "")
    date_parts = (
        message.get("published-print", {}).get("date-parts")
        or message.get("published-online", {}).get("date-parts")
        or message.get("published", {}).get("date-parts")
        or []
    )
    published = (
        "-".join(str(p) for p in date_parts[0]) if date_parts and date_parts[0] else ""
    )

    summary_bits = [title]
    if author_names:
        summary_bits.append(author_names)
    if container:
        summary_bits.append(container)
    elif publisher:
        summary_bits.append(publisher)
    if published:
        summary_bits.append(published)
    summary = " — ".join(summary_bits)[:500]

    return {
        "found": True,
        "url": url,
        "summary": summary,
        "content": json.dumps(message, default=str),
    }


def _looks_academic(claim):
    claim_lower = claim.lower()
    return any(term in claim_lower for term in _ACADEMIC_TERMS)


def _extract_quoted_title(claim):
    match = _QUOTED_RE.search(claim)
    return match.group(1) if match else None
