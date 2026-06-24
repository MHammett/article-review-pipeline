import concurrent.futures
import hashlib
import importlib
import logging

from . import wayback

log = logging.getLogger(__name__)

ADAPTER_MAP = {
    "eia": "ci_article_review.adapters.citation.sources.eia",
    "fred": "ci_article_review.adapters.citation.sources.fred",
    "census": "ci_article_review.adapters.citation.sources.census",
    "fhwa": "ci_article_review.adapters.citation.sources.fhwa",
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


def _resolve_one(claim, citation_sources):
    """Resolve a single claim against the configured sources, in order.

    Returns the resolution dict.  Pointer-only adapters (e.g. FHWA, which just
    points at a publication for manual retrieval) are reported with
    ``verification: "pointer"`` so the report does not imply the claim was
    checksummed against retrieved data.
    """
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
                return {
                    "claim": claim,
                    "source_name": source_name,
                    "url": resolved_url,
                    "content_summary": result.get("summary"),
                    "checksum": sha256_checksum(result.get("content", "")),
                    "resolved": True,
                    "verification": "pointer" if pointer_only else "checksum",
                    "wayback": wb,
                }
        except Exception as e:
            log.warning(f"Citation adapter {adapter_name} failed for claim '{claim[:50]}': {e}")

    return {
        "claim": claim,
        "resolved": False,
        "note": "No configured source adapter could resolve this claim",
    }


def resolve_citations(claims, citation_sources):
    """
    For each claim, try each configured source adapter to find a primary source.
    Claims are resolved in parallel (bounded by _MAX_PARALLEL) because each one
    performs blocking network I/O — the source lookup plus a Wayback check.
    Returns a list of resolution results in the original claim order.
    """
    if not claims:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(claims), _MAX_PARALLEL)) as pool:
        futures = {
            pool.submit(_resolve_one, claim, citation_sources): idx
            for idx, claim in enumerate(claims)
        }
        ordered: dict[int, dict] = {}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                ordered[idx] = future.result()
            except Exception as e:
                log.warning(f"Citation resolution raised for claim index {idx}: {e}")
                ordered[idx] = {
                    "claim": claims[idx],
                    "resolved": False,
                    "note": f"Resolution error: {e}",
                }

    return [ordered[i] for i in range(len(claims))]
