import concurrent.futures
import hashlib
import importlib
import logging

import requests

from ci_core.http import DEFAULT_HEADERS

from ci_core import extract
from ci_core.llm.adapters import mistral
from . import wayback

log = logging.getLogger(__name__)

#: Verdicts that pass content-relevance verification; anything else downgrades
#: the citation instead of letting it through at checksum-level confidence.
_SUPPORTING_VERDICTS = {"supports"}
_KNOWN_VERDICTS = {"supports", "contradicts", "not_addressed", "inconclusive"}

#: Below this many characters, "extracted text" is a cookie banner or a bot
#: wall, not a document. Verifying against it produces a confident-sounding
#: verdict from nothing, which is the exact failure this guard exists to stop.
_MIN_VERIFIABLE_CHARS = 200

#: Notes for the "we fetched it but could not read it" case, keyed by the kind
#: of document we failed on. Each one has to make clear that the source was not
#: judged — an honest "unverified" is fine, a wrong "does not support" is not.
_UNVERIFIABLE_NOTES = {
    "pdf": (
        "Source URL fetched and checksummed, but no text could be extracted from "
        "the PDF (scanned image with no text layer, password-protected, or pypdf "
        "not installed). Relevance was NOT assessed — this is not evidence the "
        "source fails to support the claim. Verify manually before citing."
    ),
    "html": (
        "Source URL fetched and checksummed, but no article text could be "
        "extracted (JavaScript-rendered page, paywall, or bot-block). Relevance "
        "was NOT assessed — this is not evidence the source fails to support the "
        "claim. Verify manually before citing."
    ),
    "access_wall": (
        "Source URL returned a bot-check, CAPTCHA, or paywall interstitial rather "
        "than the document itself, so the real content was never seen. Relevance "
        "was NOT assessed — this is not evidence the source fails to support the "
        "claim. Verify manually before citing."
    ),
    "default": (
        "Source URL fetched and checksummed, but produced no readable text. "
        "Relevance was NOT assessed — this is not evidence the source fails to "
        "support the claim. Verify manually before citing."
    ),
}

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


def _extract_fetched(resp, url):
    """Reduce a fetched response to readable text for checksum + verification.

    Passing ``resp.text`` straight through was the cause of near-total citation
    rejection: the first 4000 characters of any real page are doctype, <head>,
    meta/script/link tags and nav, so the verifier was asked whether markup
    supported a claim and correctly said no. Raw PDF bytes were worse — binary
    handed to a text model.

    Returns ``(text, kind)``; an empty ``text`` means the body could not be
    read, which the caller must report as unverifiable rather than as a
    content mismatch.
    """
    return extract.extract_response_text(
        resp.content,
        content_type=resp.headers.get("Content-Type"),
        url=url,
        encoding=resp.encoding or "utf-8",
    )


def _wayback_fallback_content(url, timeout):
    """After a direct 403, try archive.org's snapshot as the content source.

    archive.org serves its own cached copy, so a site blocking our fetch
    doesn't block the archived one. A single attempt, no retry loop. Returns
    the snapshot's ``(url, text, kind)`` on success, or None if there's no
    snapshot or the snapshot itself doesn't resolve. The snapshot body goes
    through the same extraction as a direct fetch — a Wayback page is HTML
    (with archive.org's own banner chrome on top), so it needs it even more.
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
    text, kind = _extract_fetched(resp, snapshot_url)
    return snapshot_url, text, kind


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

    # Centre the excerpt on the passage most likely to address the claim.
    # Blind head+tail truncation cuts the supporting sentence out of any long
    # document — the limit table in a 60-page guidelines PDF is never in the
    # first 4000 characters.
    excerpt = extract.select_excerpt(content, claim, head=4000, tail=1000)
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


def _resolve_known_url(claim, known_url, api_keys=None, call_log=None, timeout=15):
    """Resolve a claim whose source URL is already known (e.g. supplied by the
    fact-check model itself), bypassing the narrow adapter matching entirely.

    A direct 403 (but not a 404 or 5xx — see ``_wayback_fallback_content``'s
    scoping) triggers a single fallback attempt against a Wayback snapshot of
    the same URL, so a claim isn't reported unresolved just because the origin
    site blocks automated fetches.

    The fetched body is reduced to readable text (main-article extraction for
    HTML, pypdf for PDFs) before anything else touches it — see
    ``_extract_fetched``. If nothing readable comes out, the citation is marked
    ``unverifiable`` rather than run through verification, because a verdict
    derived from markup or binary asserts something false about the source.

    Otherwise the extracted text is passed through ``_verify_relevance`` before
    the citation is allowed to carry the strongest "checksum" verification tier
    — see that function's docstring for why a loaded URL alone isn't proof the
    claim is supported.
    """
    verified_via = "direct"
    try:
        resp = requests.get(known_url, timeout=timeout, headers=DEFAULT_HEADERS)
        resp.raise_for_status()
        content, content_kind = _extract_fetched(resp, known_url)
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
        _, content, content_kind = fallback
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
        "content_kind": content_kind,
        "wayback": wb,
    }

    if extract.looks_like_access_wall(content):
        # A CAPTCHA/paywall interstitial served as HTTP 200. It extracts into
        # clean prose, so only this check stops the verifier from reading the
        # blocking notice and reporting that the *source* fails the claim.
        result["verification"] = "unverifiable"
        result["content_kind"] = "access_wall"
        result["note"] = _UNVERIFIABLE_NOTES["access_wall"]
        return result

    if len(content.strip()) < _MIN_VERIFIABLE_CHARS:
        # We fetched a real document but could not read it: a PDF with no text
        # layer (or no pypdf installed), a JS-only render, a bot wall. Say that
        # — do NOT run verification and report "does not support the claim",
        # which asserts something false about a source we never read.
        result["verification"] = "unverifiable"
        result["note"] = _UNVERIFIABLE_NOTES.get(
            content_kind, _UNVERIFIABLE_NOTES["default"]
        )
        return result

    verdict_info, verification_call_log = _verify_relevance(claim, content, api_keys)
    if call_log is not None and verification_call_log is not None:
        call_log.append(verification_call_log)

    if not verdict_info["checked"]:
        # The check itself didn't run or couldn't be interpreted (no API key,
        # call failure, unparseable verdict). We read the page but formed no
        # opinion on it, so this is "could not assess", not "checksum-verified"
        # and emphatically not "does not support".
        result["verification"] = "unverifiable"
        result["relevance_check"] = verdict_info["reason"]
        result["note"] = (
            "Source URL fetched and checksummed, but relevance could not be "
            f"assessed: {verdict_info['reason']}. This is NOT evidence the source "
            "fails to support the claim — verify manually before citing."
        )
        return result

    result["relevance_verdict"] = verdict_info["verdict"]
    result["relevance_reason"] = verdict_info["reason"]
    if verdict_info["verdict"] not in _SUPPORTING_VERDICTS:
        # The URL loaded and checksummed fine, but content verification says
        # it doesn't actually back the claim — never let that pass silently
        # at checksum-level confidence.
        result["resolved"] = False
        result["verification"] = "content_mismatch"
        result["note"] = (
            f"Source URL loaded, and its extracted article text was read and "
            f"checked, but content verification found it does not support this "
            f"specific claim ({verdict_info['verdict']}): {verdict_info['reason']}"
        )

    return result


def _resolve_one(claim, citation_sources, known_url=None, api_keys=None, call_log=None):
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
        return _resolve_known_url(
            claim, known_url, api_keys=api_keys, call_log=call_log
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


def resolve_citations(
    claims, citation_sources, api_keys=None, verification_call_log=None
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

    Returns a list of resolution results in the original claim order.
    """
    if not claims:
        return []

    normalized = [_normalize_claim_entry(entry) for entry in claims]
    call_log = verification_call_log if verification_call_log is not None else []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(normalized), _MAX_PARALLEL)
    ) as pool:
        futures = {
            pool.submit(
                _resolve_one, claim, citation_sources, known_url, api_keys, call_log
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
