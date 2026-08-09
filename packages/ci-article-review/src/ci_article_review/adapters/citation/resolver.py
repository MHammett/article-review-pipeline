import concurrent.futures
import hashlib
import importlib
import logging

import requests

from ci_core.http import DEFAULT_HEADERS

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

#: Bound on concurrent Wayback *submissions*, kept well below _MAX_PARALLEL.
#: archive.org's Save Page Now API does a real page capture (slow, rate-limited),
#: not a cheap read, so a burst of 8 concurrent submissions risks getting the
#: pipeline's IP rate-limited or blocked. Submissions run as a follow-up pass
#: after the main resolution completes, rather than inline per-claim.
_MAX_SUBMIT_PARALLEL = 2


def _load_adapter(adapter_name):
    module_path = ADAPTER_MAP.get(adapter_name)
    if not module_path:
        raise ValueError(f"Unknown citation adapter: {adapter_name}")
    return importlib.import_module(module_path)


def sha256_checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _wayback_fallback_content(url, timeout):
    """After a direct 403, try archive.org's snapshot as the content source.

    archive.org serves its own cached copy, so a site blocking our fetch
    doesn't block the archived one. A single attempt, no retry loop. Returns
    the snapshot's (url, text) on success, or None if there's no snapshot or
    the snapshot itself doesn't resolve.
    """
    wb = wayback.check(url, timeout=timeout)
    snapshot_url = wb.get("snapshot_url")
    if not wb.get("archived") or not snapshot_url:
        return None
    try:
        resp = requests.get(snapshot_url, timeout=timeout, headers=DEFAULT_HEADERS)
        resp.raise_for_status()
    except Exception:
        return None
    return snapshot_url, resp.text


def _resolve_known_url(claim, known_url, timeout=15):
    """Resolve a claim whose source URL is already known (e.g. supplied by the
    fact-check model itself), bypassing the narrow adapter matching entirely.

    A direct 403 (but not a 404 or 5xx — see ``_wayback_fallback_content``'s
    scoping) triggers a single fallback attempt against a Wayback snapshot of
    the same URL, so a claim isn't reported unresolved just because the origin
    site blocks automated fetches.
    """
    verified_via = "direct"
    try:
        resp = requests.get(known_url, timeout=timeout, headers=DEFAULT_HEADERS)
        resp.raise_for_status()
        content = resp.text
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        fallback = (
            _wayback_fallback_content(known_url, timeout) if status == 403 else None
        )
        if fallback is None:
            log.warning(f"Known source URL fetch failed for claim '{claim[:50]}': {e}")
            return {
                "claim": claim,
                "url": known_url,
                "resolved": False,
                "note": f"Known source URL could not be fetched: {e}",
            }
        _, content = fallback
        verified_via = "wayback_fallback"
    except Exception as e:
        log.warning(f"Known source URL fetch failed for claim '{claim[:50]}': {e}")
        return {
            "claim": claim,
            "url": known_url,
            "resolved": False,
            "note": f"Known source URL could not be fetched: {e}",
        }

    wb = wayback.check(known_url)
    return {
        "claim": claim,
        "source_name": "fact-check model",
        "url": known_url,
        "content_summary": content[:500],
        "checksum": sha256_checksum(content),
        "resolved": True,
        "verification": "checksum",
        "verified_via": verified_via,
        "wayback": wb,
    }


def _resolve_one(claim, citation_sources, known_url=None):
    """Resolve a single claim against the configured sources, in order.

    If ``known_url`` is given (a source URL the fact-check model already
    supplied for this claim), it is fetched and checksummed directly —
    the narrow adapter matching below is skipped entirely.

    Returns the resolution dict.  Pointer-only adapters (e.g. FHWA, which just
    points at a publication for manual retrieval) are reported with
    ``verification: "pointer"`` so the report does not imply the claim was
    checksummed against retrieved data.
    """
    if known_url:
        return _resolve_known_url(claim, known_url)

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
            log.warning(
                f"Citation adapter {adapter_name} failed for claim '{claim[:50]}': {e}"
            )

    return {
        "claim": claim,
        "resolved": False,
        "note": "No configured source adapter could resolve this claim",
    }


def _submit_missing_archives(results, archive_org_creds=None):
    """Follow-up pass: request Wayback archiving for resolved citations whose
    URL isn't archived yet.

    Runs after the main resolution pass, at a lower concurrency
    (_MAX_SUBMIT_PARALLEL) than claim resolution — archive.org's Save Page Now
    API does a real page capture per request, so submitting inline per-claim
    at the same parallelism as resolution risks a rate-limit or IP block.
    Submission is fire-and-forget: it does not verify the capture completed
    (that can take seconds to minutes on archive.org's side); a future run's
    ``wayback.check()`` will pick up the new snapshot once it exists.

    Mutates each result's ``wayback`` dict in place, adding ``submitted`` and,
    on failure, ``submission_error``. Never raises — a submission failure
    degrades to "still shows as unarchived", it does not fail the run.
    """
    creds = archive_org_creds or {}
    access_key = creds.get("access_key")
    secret_key = creds.get("secret_key")

    targets = [
        r
        for r in results
        if r.get("resolved")
        and r.get("url")
        and r.get("wayback", {}).get("archived") is False
    ]
    if not targets:
        return

    def _submit_one(entry):
        try:
            sub = wayback.submit(
                entry["url"], access_key=access_key, secret_key=secret_key
            )
        except Exception as e:
            log.warning(f"Wayback submission raised for {entry['url']}: {e}")
            sub = {"submitted": False, "error": str(e)}
        entry["wayback"]["submitted"] = sub.get("submitted", False)
        if sub.get("error"):
            entry["wayback"]["submission_error"] = sub["error"]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(targets), _MAX_SUBMIT_PARALLEL)
    ) as pool:
        list(pool.map(_submit_one, targets))


def _normalize_claim_entry(entry):
    """Accept either a plain claim string or a dict with an optional
    already-known source URL: {"claim": str, "known_url": str | None}.
    """
    if isinstance(entry, str):
        return entry, None
    return entry.get("claim", ""), entry.get("known_url")


def resolve_citations(claims, citation_sources, api_keys=None):
    """
    For each claim, resolve a primary source. If the claim entry carries a
    ``known_url`` (a source the fact-check model already supplied), that URL
    is fetched and checksummed directly. Otherwise each configured source
    adapter is tried in order. Claims are resolved in parallel (bounded by
    _MAX_PARALLEL) because each one performs blocking network I/O — the
    source lookup plus a Wayback check.

    After resolution, any resolved citation whose URL isn't yet archived is
    submitted to Wayback's Save Page Now API for archiving (see
    ``_submit_missing_archives``). Optional ``api_keys["archive_org"]``
    (``{"access_key": ..., "secret_key": ...}``) authenticates those
    submissions; without it, submission falls back to the unauthenticated
    trigger endpoint.

    Returns a list of resolution results in the original claim order.
    """
    if not claims:
        return []

    normalized = [_normalize_claim_entry(entry) for entry in claims]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(normalized), _MAX_PARALLEL)
    ) as pool:
        futures = {
            pool.submit(_resolve_one, claim, citation_sources, known_url): idx
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

    resolved_results = [ordered[i] for i in range(len(normalized))]
    _submit_missing_archives(resolved_results, (api_keys or {}).get("archive_org"))
    return resolved_results
