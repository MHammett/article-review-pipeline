import concurrent.futures
import hashlib
import importlib
import logging

import requests

from ci_core.http import USER_AGENT

from ci_article_review import history_analytics

from . import wayback

log = logging.getLogger(__name__)

ADAPTER_MAP = {
    "eia": "ci_article_review.adapters.citation.sources.eia",
    "fred": "ci_article_review.adapters.citation.sources.fred",
    "census": "ci_article_review.adapters.citation.sources.census",
    "fhwa": "ci_article_review.adapters.citation.sources.fhwa",
    "crossref": "ci_article_review.adapters.citation.sources.crossref",
    "epa": "ci_article_review.adapters.citation.sources.epa",
    "pjm": "ci_article_review.adapters.citation.sources.pjm",
    "icc": "ci_article_review.adapters.citation.sources.icc",
    "ferc": "ci_article_review.adapters.citation.sources.ferc",
    "ilga": "ci_article_review.adapters.citation.sources.ilga",
}

#: Bound on concurrent claim resolutions so we don't open dozens of sockets at once.
_MAX_PARALLEL = 8


def _load_adapter(adapter_name):
    module_path = ADAPTER_MAP.get(adapter_name)
    if not module_path:
        raise ValueError(f"Unknown citation adapter: {adapter_name}")
    return importlib.import_module(module_path)


def sha256_checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_checksum_index(history_root=None):
    """url -> most recent prior {checksum, article_slug, run_number, generated}.

    Scans every report under ``history_root`` (any article, any run — sources
    like EPA eGRID or EIA state profiles get cited across many different
    articles for the same publication) once per pipeline run, so resolving
    individual claims can do an O(1) dict lookup instead of re-scanning
    history on every claim.
    """
    if history_root is None:
        history_root = history_analytics.HISTORY_ROOT

    index = {}
    for entry in history_analytics.load_reports(history_root):
        citations = entry["report"].get("section_9_citations") or []
        for c in citations:
            url = c.get("url")
            checksum = c.get("checksum")
            if not url or not checksum:
                continue
            index[url] = {
                "checksum": checksum,
                "article_slug": entry["slug"],
                "run_number": entry["report"].get("run_number"),
                "generated": entry["report"].get("generated"),
            }
    return index


def _check_drift(result, checksum_index):
    """If ``result``'s URL was checksummed in a prior run with a different
    value, annotate the resolution with a ``content_changed_since`` field.

    A source legitimately changing over time (e.g. an annual data report's
    yearly update) isn't itself an error — this is a signal to surface to a
    human reviewer, not a resolution failure, so it never raises or blocks.
    """
    url = result.get("url")
    checksum = result.get("checksum")
    if not url or not checksum:
        return result

    prior = checksum_index.get(url)
    if prior is None or prior["checksum"] == checksum:
        return result

    prior_run = prior["run_number"]
    prior_article = prior["article_slug"]
    prior_date = prior["generated"]
    when = f" on {prior_date}" if prior_date else ""
    result["content_changed_since"] = {
        "prior_checksum": prior["checksum"],
        "prior_run": prior_run,
        "prior_article": prior_article,
        "prior_date": prior_date,
        "note": (
            "This source's content differs from the last time it was "
            f"checksummed (run {prior_run} of '{prior_article}'{when}). "
            "A claim previously verified against it may need re-checking."
        ),
    }
    return result


def _resolve_known_url(claim, known_url, checksum_index, timeout=15):
    """Resolve a claim whose source URL is already known (e.g. supplied by the
    fact-check model itself), bypassing the narrow adapter matching entirely.
    """
    try:
        resp = requests.get(
            known_url, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
        resp.raise_for_status()
        content = resp.text
    except Exception as e:
        log.warning(f"Known source URL fetch failed for claim '{claim[:50]}': {e}")
        return {
            "claim": claim,
            "url": known_url,
            "resolved": False,
            "note": f"Known source URL could not be fetched: {e}",
        }

    wb = wayback.check(known_url)
    return _check_drift(
        {
            "claim": claim,
            "source_name": "fact-check model",
            "url": known_url,
            "content_summary": content[:500],
            "checksum": sha256_checksum(content),
            "resolved": True,
            "verification": "checksum",
            "wayback": wb,
        },
        checksum_index,
    )


def _resolve_one(claim, citation_sources, known_url=None, checksum_index=None):
    """Resolve a single claim against the configured sources, in order.

    If ``known_url`` is given (a source URL the fact-check model already
    supplied for this claim), it is fetched and checksummed directly —
    the narrow adapter matching below is skipped entirely.

    Returns the resolution dict.  Pointer-only adapters (e.g. FHWA, which just
    points at a publication for manual retrieval) are reported with
    ``verification: "pointer"`` so the report does not imply the claim was
    checksummed against retrieved data.
    """
    if checksum_index is None:
        checksum_index = {}

    if known_url:
        return _resolve_known_url(claim, known_url, checksum_index)

    for source_config in citation_sources:
        adapter_name = source_config.get("adapter")
        if not adapter_name or adapter_name == "generic_url":
            continue
        source_name = source_config.get("name", adapter_name)
        try:
            adapter = _load_adapter(adapter_name)
            result = adapter.resolve(claim)
            if result and result.get("found"):
                resolved_url = result.get("url")
                wb = wayback.check(resolved_url) if resolved_url else {"archived": None}
                pointer_only = bool(result.get("pointer_only"))
                return _check_drift(
                    {
                        "claim": claim,
                        "source_name": source_name,
                        "url": resolved_url,
                        "content_summary": result.get("summary"),
                        "checksum": sha256_checksum(result.get("content", "")),
                        "resolved": True,
                        "verification": "pointer" if pointer_only else "checksum",
                        "wayback": wb,
                    },
                    checksum_index,
                )
        except Exception as e:
            log.warning(
                f"Citation adapter {adapter_name} failed for claim '{claim[:50]}': {e}"
            )

    return {
        "claim": claim,
        "resolved": False,
        "note": "No configured source adapter could resolve this claim",
    }


def _normalize_claim_entry(entry):
    """Accept either a plain claim string or a dict with an optional
    already-known source URL: {"claim": str, "known_url": str | None}.
    """
    if isinstance(entry, str):
        return entry, None
    return entry.get("claim", ""), entry.get("known_url")


def resolve_citations(claims, citation_sources, history_root=None):
    """
    For each claim, resolve a primary source. If the claim entry carries a
    ``known_url`` (a source the fact-check model already supplied), that URL
    is fetched and checksummed directly. Otherwise each configured source
    adapter is tried in order. Claims are resolved in parallel (bounded by
    _MAX_PARALLEL) because each one performs blocking network I/O — the
    source lookup plus a Wayback check.

    Before resolving, a URL -> prior-checksum index is built once from
    ``history_root`` (every run, every article) so each resolved URL can be
    compared against the last time it was checksummed — see
    ``build_checksum_index`` / ``_check_drift``.

    Returns a list of resolution results in the original claim order.
    """
    if not claims:
        return []

    normalized = [_normalize_claim_entry(entry) for entry in claims]
    checksum_index = build_checksum_index(history_root)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(normalized), _MAX_PARALLEL)
    ) as pool:
        futures = {
            pool.submit(
                _resolve_one, claim, citation_sources, known_url, checksum_index
            ): idx
            for idx, (claim, known_url) in enumerate(normalized)
        }
        ordered: dict[int, dict] = {}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                ordered[idx] = future.result()
            except Exception as e:
                log.warning(f"Citation resolution raised for claim index {idx}: {e}")
                ordered[idx] = {
                    "claim": normalized[idx][0],
                    "resolved": False,
                    "note": f"Resolution error: {e}",
                }

    return [ordered[i] for i in range(len(normalized))]
