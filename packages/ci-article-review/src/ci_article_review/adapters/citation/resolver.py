import concurrent.futures
import hashlib
import importlib
import logging

import requests

from ci_core.http import DEFAULT_HEADERS

from ci_core import redact
from ci_core.llm.adapters import mistral

from ci_article_review import history_analytics

from . import wayback

log = logging.getLogger(__name__)

#: Verdicts that pass content-relevance verification; anything else downgrades
#: the citation instead of letting it through at checksum-level confidence.
_SUPPORTING_VERDICTS = {"supports"}
_KNOWN_VERDICTS = {"supports", "contradicts", "not_addressed", "inconclusive"}

#: Cheap/fast model used for the relevance check — this runs once per known_url
#: citation, so it deliberately avoids a heavyweight reasoning model.
_VERIFICATION_MODEL = "mistral-small-latest"

_VERIFICATION_SYSTEM_PROMPT = (
    "You verify whether a web page's content supports a specific factual claim. "
    "Respond with ONLY a JSON object of the form "
    '{"verdict": "supports" | "contradicts" | "not_addressed" | "inconclusive", '
    '"reason": "<one sentence>"}. '
    '"supports" means the page content directly backs the claim. "contradicts" means '
    'the page says something that conflicts with the claim. "not_addressed" means the '
    'page does not discuss this claim at all. "inconclusive" means the page is related '
    "but too ambiguous to judge either way."
)

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


def build_checksum_index(history_root=None):
    """url -> most recent prior {checksum, article_slug, run_number, generated}.

    Scans every report under ``history_root`` (any article, any run — sources
    like EPA eGRID or EIA state profiles get cited across many different
    articles for the same publication) once per pipeline run, so resolving
    individual claims can do an O(1) dict lookup instead of re-scanning
    history on every claim.

    Only ``verification: "checksum"`` entries are indexed. A pointer-only
    entry's checksum is taken over whatever the adapter returned as content —
    usually nothing at all — so indexing those would mean a URL first cited
    pointer-only and later fetched for real reads as "changed" every time,
    which is a property of the two tiers differing, not of the page moving.
    Reports predating the ``verification`` field are skipped for the same
    reason: their tier can't be established, so no comparison is trustworthy.
    """
    if history_root is None:
        history_root = history_analytics.HISTORY_ROOT

    index = {}
    for entry in history_analytics.load_reports(history_root):
        citations = entry["report"].get("section_9_citations") or []
        for c in citations:
            url = c.get("url")
            checksum = c.get("checksum")
            if not url or not checksum or c.get("verification") != "checksum":
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

    Only applies to the ``checksum`` tier — that's the only tier where the
    checksum is taken over content actually retrieved as evidence, so it's
    the only tier where a difference says something about the source.

    A source legitimately changing over time (an annual data report's yearly
    update, but also a rotating ad slot or a rendered timestamp) isn't itself
    an error — this is a signal to surface to a human reviewer, not a
    resolution failure, so it never raises or blocks.
    """
    if result.get("verification") != "checksum":
        return result

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


def _verify_relevance(claim, content, api_keys):
    """Ask a cheap/fast model whether ``content`` actually supports ``claim``.

    ``known_url`` citations often come from an ungrounded model recalling a
    URL from training data rather than a live search — the URL loading is not
    evidence the page says what the claim says. This makes a real check.

    Returns ``(verdict_info, call_log_entry)``. ``verdict_info`` is
    ``{"checked": False, "reason": str}`` when the check couldn't run (no
    credentials, call failure, unparseable verdict) or
    ``{"checked": True, "verdict": str, "reason": str}`` on success.
    ``call_log_entry`` is a cost-tracking dict (same shape as pipeline.py's
    ``api_call_log`` entries) when a call was actually attempted, else None.
    Never raises — a failure here must degrade to the pre-existing unverified
    behavior, not crash citation resolution.
    """
    api_key = ((api_keys or {}).get("mistral") or {}).get("api_key", "")
    if not api_key:
        return {
            "checked": False,
            "reason": "relevance check skipped: no mistral API key configured",
        }, None

    excerpt = redact.truncate_excerpt(content, head=4000, tail=1000)
    user_prompt = f'Claim: "{claim}"\n\nPage content:\n{excerpt}'

    try:
        result = mistral.call(
            _VERIFICATION_SYSTEM_PROMPT,
            user_prompt,
            api_key,
            model=_VERIFICATION_MODEL,
        )
    except Exception as e:
        log.warning(
            f"Citation relevance verification raised for claim '{claim[:50]}': {e}"
        )
        return {"checked": False, "reason": f"relevance check failed: {e}"}, None

    call_log_entry = {
        "pass": "citation_verification:known_url",
        "model": result.get("model", _VERIFICATION_MODEL),
        "failed": bool(result.get("failed")),
        "tokens": result.get("tokens", {}),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "error": result.get("error") if result.get("failed") else None,
    }

    if result.get("failed"):
        return {
            "checked": False,
            "reason": f"relevance check call failed: {result.get('error', 'unknown error')}",
        }, call_log_entry

    data = result.get("data") or {}
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in _KNOWN_VERDICTS:
        return {
            "checked": False,
            "reason": f"relevance check returned an unrecognized verdict: {verdict!r}",
        }, call_log_entry

    return {
        "checked": True,
        "verdict": verdict,
        "reason": data.get("reason", ""),
    }, call_log_entry


def _resolve_known_url(
    claim, known_url, api_keys=None, call_log=None, checksum_index=None, timeout=15
):
    """Resolve a claim whose source URL is already known (e.g. supplied by the
    fact-check model itself), bypassing the narrow adapter matching entirely.

    A direct 403 (but not a 404 or 5xx — see ``_wayback_fallback_content``'s
    scoping) triggers a single fallback attempt against a Wayback snapshot of
    the same URL, so a claim isn't reported unresolved just because the origin
    site blocks automated fetches.

    After a successful fetch, the content is passed through
    ``_verify_relevance`` before the citation is allowed to carry the
    strongest "checksum" verification tier — see that function's docstring
    for why a loaded URL alone isn't proof the claim is supported.
    """
    checksum_index = checksum_index if checksum_index is not None else {}
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
    result = {
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

    verdict_info, verification_call_log = _verify_relevance(claim, content, api_keys)
    if call_log is not None and verification_call_log is not None:
        call_log.append(verification_call_log)

    if not verdict_info["checked"]:
        # Verification didn't run or couldn't be interpreted — degrade
        # gracefully to the pre-existing (unverified-relevance) behavior
        # rather than blocking resolution, but leave a clear note.
        result["relevance_check"] = verdict_info["reason"]
        return _check_drift(result, checksum_index)

    result["relevance_verdict"] = verdict_info["verdict"]
    result["relevance_reason"] = verdict_info["reason"]
    if verdict_info["verdict"] not in _SUPPORTING_VERDICTS:
        # The URL loaded and checksummed fine, but content verification says
        # it doesn't actually back the claim — never let that pass silently
        # at checksum-level confidence.
        result["resolved"] = False
        result["verification"] = "content_mismatch"
        result["note"] = (
            f"Source URL loaded and checksummed, but content verification found "
            f"it does not support this specific claim "
            f"({verdict_info['verdict']}): {verdict_info['reason']}"
        )

    return _check_drift(result, checksum_index)


def _resolve_one(
    claim,
    citation_sources,
    known_url=None,
    api_keys=None,
    call_log=None,
    checksum_index=None,
):
    """Resolve a single claim against the configured sources, in order.

    If ``known_url`` is given (a source URL the fact-check model already
    supplied for this claim), it is fetched and checksummed directly —
    the narrow adapter matching below is skipped entirely.

    Returns the resolution dict.  Pointer-only adapters (e.g. FHWA, which just
    points at a publication for manual retrieval) are reported with
    ``verification: "pointer"`` so the report does not imply the claim was
    checksummed against retrieved data.
    """
    checksum_index = checksum_index if checksum_index is not None else {}

    if known_url:
        return _resolve_known_url(
            claim,
            known_url,
            api_keys=api_keys,
            call_log=call_log,
            checksum_index=checksum_index,
        )

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


def resolve_citations(
    claims,
    citation_sources,
    api_keys=None,
    verification_call_log=None,
    history_root=None,
):
    """
    For each claim, resolve a primary source. If the claim entry carries a
    ``known_url`` (a source the fact-check model already supplied), that URL
    is fetched and checksummed directly. Otherwise each configured source
    adapter is tried in order. Claims are resolved in parallel (bounded by
    _MAX_PARALLEL) because each one performs blocking network I/O — the
    source lookup plus a Wayback check.

    A ``known_url`` fetch also runs a content-relevance check (see
    ``_verify_relevance``) — a cheap LLM call confirming the fetched page
    actually supports the claim, not just that the URL loaded. Passing a
    ``verification_call_log`` list (e.g. the pipeline's ``api_call_log``)
    appends one cost-tracking entry per relevance check performed, in the
    same shape as the pipeline's own entries, so it flows into
    ``cost_analysis.calculate`` like any other model call. Without it, the
    checks still run but their cost isn't tracked anywhere.

    After resolution, any resolved citation whose URL isn't yet archived is
    submitted to Wayback's Save Page Now API for archiving (see
    ``_submit_missing_archives``). Optional ``api_keys["archive_org"]``
    (``{"access_key": ..., "secret_key": ...}``) authenticates those
    submissions; without it, submission falls back to the unauthenticated
    trigger endpoint.

    Before resolving, a URL -> prior-checksum index is built once from
    ``history_root`` (every run, every article) so each citation resolved at
    the ``checksum`` tier can be compared against the last time that URL was
    checksummed — see ``build_checksum_index`` / ``_check_drift``. Passing no
    ``history_root`` skips the comparison entirely rather than defaulting to
    the on-disk history, so a caller that doesn't opt in never pays for a
    history scan.

    Returns a list of resolution results in the original claim order.
    """
    if not claims:
        return []

    normalized = [_normalize_claim_entry(entry) for entry in claims]
    call_log = verification_call_log if verification_call_log is not None else []
    checksum_index = build_checksum_index(history_root) if history_root else {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(normalized), _MAX_PARALLEL)
    ) as pool:
        futures = {
            pool.submit(
                _resolve_one,
                claim,
                citation_sources,
                known_url,
                api_keys,
                call_log,
                checksum_index,
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

    resolved_results = [ordered[i] for i in range(len(normalized))]
    _submit_missing_archives(resolved_results, (api_keys or {}).get("archive_org"))
    return resolved_results
