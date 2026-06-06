import hashlib
import importlib
import logging
import re

log = logging.getLogger(__name__)

ADAPTER_MAP = {
    "eia": "adapters.citation.sources.eia",
    "fred": "adapters.citation.sources.fred",
    "census": "adapters.citation.sources.census",
    "fhwa": "adapters.citation.sources.fhwa",
}


def _load_adapter(adapter_name):
    module_path = ADAPTER_MAP.get(adapter_name)
    if not module_path:
        raise ValueError(f"Unknown citation adapter: {adapter_name}")
    return importlib.import_module(module_path)


def sha256_checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def resolve_citations(claims, citation_sources):
    """
    For each claim, try each configured source adapter to find a primary source.
    Returns list of resolution results.
    """
    results = []
    for claim in claims:
        resolved = False
        for source_config in citation_sources:
            adapter_name = source_config.get("adapter")
            if not adapter_name or adapter_name == "generic_url":
                continue
            try:
                adapter = _load_adapter(adapter_name)
                result = adapter.resolve(claim)
                if result and result.get("found"):
                    checksum = sha256_checksum(result.get("content", ""))
                    results.append({
                        "claim": claim,
                        "source_name": source_config["name"],
                        "url": result.get("url"),
                        "content_summary": result.get("summary"),
                        "checksum": checksum,
                        "resolved": True,
                    })
                    resolved = True
                    break
            except Exception as e:
                log.warning(f"Citation adapter {adapter_name} failed for claim '{claim[:50]}': {e}")

        if not resolved:
            results.append({
                "claim": claim,
                "resolved": False,
                "note": "No configured source adapter could resolve this claim",
            })

    return results
