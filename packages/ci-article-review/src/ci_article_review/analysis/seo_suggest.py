"""Propose SEO metadata for an article under review — one cheap model call.

``analysis.seo`` validates SEO and reports what is missing. Nothing proposed
anything, which left two holes. The publication handoff template offers
"derive from primary claim" / "derive from opening paragraph" as SEO METADATA
values, and ``handoff_parser`` even guards against those placeholders reaching
WordPress un-derived — but no code ever derived them. And the draft submission
template has no SEO section at all, so ``no_meta_description`` fired on every
draft run against a field that format cannot supply.

This module fills both: it drafts the values, at draft-review time, where they
feed the chat revision round-trip and where seeing the intended keyword early
can reveal that the article never actually uses the phrase it should rank for.

Everything here is advisory. Keyword choice is strategic — what the author
wants to rank for — so candidates are returned for a human to pick from, and
nothing is written into any config, handoff, or WordPress metadata.

The call follows ``adapters/citation/resolver.py``'s ``_verify_relevance``: a
small fast model, one call, cost logged under its own pass name, and any
failure degrading to "no suggestions this run" rather than failing the run.
"""

import logging
import re

from ci_core import redact
from ci_core.llm.adapters import mistral

from . import seo as seo_analysis

log = logging.getLogger(__name__)

#: Cheap/fast model — this runs once per review and produces a few short
#: strings, so it deliberately avoids a heavyweight reasoning model. Same
#: choice, and same reasoning, as the citation relevance verifier.
_SUGGESTION_MODEL = "mistral-small-latest"

#: Cost-summary pass name. Distinct from the review passes (``model:domain``)
#: so the suggestion's cost is attributable at a glance.
CALL_LOG_PASS = "seo_suggestions"

#: How much of the article body to send. The opening carries the framing a
#: meta description has to summarize and the tail usually restates the thesis;
#: the heading outline (sent separately, in full) covers the middle, which is
#: why a blind head+tail of a 15,000-word piece isn't what goes over the wire.
_BODY_HEAD_CHARS = 3000
_BODY_TAIL_CHARS = 500

#: Cap on outline entries — a heavily-subheaded piece shouldn't crowd out the
#: body text in the prompt.
_MAX_OUTLINE_ENTRIES = 40

#: Model is asked for 3-5; this is the hard cap applied to whatever comes back.
_MAX_KEYWORD_CANDIDATES = 5

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

_SYSTEM_PROMPT = (
    "You propose SEO metadata options for an article that is still being "
    "reviewed, before its author has chosen any. You do not decide anything — "
    "the author picks from what you offer.\n"
    "Respond with ONLY a JSON object of the form "
    '{"keyword_candidates": [{"keyword": "<focus keyword or key phrase>", '
    '"rationale": "<one line>"}], "meta_description": "<one sentence or two>", '
    '"og_title": "<shorter title>"}.\n'
    "Rules:\n"
    "- Give 3 to 5 keyword candidates, strongest first. Each must be a phrase "
    "a real reader would type into a search engine, specific to what THIS "
    "article establishes — not a generic topic label for the subject area.\n"
    "- Each rationale is ONE line covering what the phrase would rank for and "
    "who is searching it. If the article never actually uses the phrase, say "
    "so in the rationale — that is useful to the author, not a reason to drop "
    "the candidate.\n"
    "- The meta description must read as prose that describes what the article "
    "establishes. It is not a keyword list, and it must not oversell: no "
    '"everything you need to know", no invented findings.\n'
    "- Omit the og_title key entirely unless the prompt explicitly asks for one."
)


def _outline(text):
    """The article's H1/H2/H3 outline, in document order, as markdown lines."""
    headings = _HEADING_RE.findall(text or "")
    lines = [f"{hashes} {title.strip()}" for hashes, title in headings]
    if len(lines) > _MAX_OUTLINE_ENTRIES:
        omitted = len(lines) - _MAX_OUTLINE_ENTRIES
        lines = lines[:_MAX_OUTLINE_ENTRIES] + [f"...[{omitted} more heading(s)]"]
    return "\n".join(lines)


def _build_user_prompt(text, handoff, pub_config, meta_limit, og_title_request):
    handoff = handoff or {}
    pub_config = pub_config or {}

    parts = [f"ARTICLE TITLE: {handoff.get('title', '')}"]
    if handoff.get("primary_claim"):
        parts.append(f"PRIMARY CLAIM: {handoff['primary_claim']}")
    if handoff.get("target_audience"):
        parts.append(f"TARGET AUDIENCE: {handoff['target_audience']}")
    if pub_config.get("publication_description"):
        parts.append(f"PUBLICATION: {pub_config['publication_description']}")
    if pub_config.get("audience"):
        parts.append(f"PUBLICATION AUDIENCE: {pub_config['audience']}")

    outline = _outline(text)
    if outline:
        parts.append(f"HEADING OUTLINE:\n{outline}")

    excerpt = redact.truncate_excerpt(
        text or "", head=_BODY_HEAD_CHARS, tail=_BODY_TAIL_CHARS
    )
    parts.append(
        f"ARTICLE TEXT (opening and close; the outline above covers the rest):\n{excerpt}"
    )

    parts.append(
        f"\nThe meta description must be under {meta_limit} characters — count them."
    )
    if og_title_request:
        parts.append(og_title_request)
    return "\n\n".join(parts)


def _clean_candidates(raw):
    """Normalize whatever came back into ``[{keyword, rationale}]``, capped.

    Tolerant of a model that returns bare strings instead of objects: a
    keyword with no rationale is still a usable candidate, and dropping it
    over shape would waste the call.
    """
    candidates = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            keyword, rationale = item, ""
        elif isinstance(item, dict):
            keyword = item.get("keyword") or item.get("phrase") or ""
            rationale = item.get("rationale") or item.get("reason") or ""
        else:
            continue
        keyword = str(keyword).strip()
        if not keyword:
            continue
        candidates.append({"keyword": keyword, "rationale": str(rationale).strip()})
        if len(candidates) == _MAX_KEYWORD_CANDIDATES:
            break
    return candidates


def _measured(value, limit):
    """``(text, chars, over_limit)`` for a length-constrained suggestion.

    An over-limit value is reported, not silently truncated and not silently
    dropped. Cutting at a word boundary produces a dangling clause that reads
    worse than the author trimming it themselves, and dropping it throws away
    the only draft the run produced — so the length is stated and flagged, and
    the author decides.
    """
    text = str(value or "").strip()
    return text, len(text), len(text) > limit


def _skipped(reason):
    log.info("SEO suggestions: %s", reason)
    return {"status": "skipped", "reason": reason}, None


def generate(text, handoff=None, pub_config=None, api_keys=None, seo_result=None):
    """Draft SEO metadata for ``text``. Returns ``(suggestions, call_log_entry)``.

    ``seo_result`` is the dict from ``seo.analyze`` for the same article; an
    OG title is only requested when it flagged the title as too long, since
    that is the case OG title exists to solve.

    ``suggestions`` always has a ``status``: ``ok`` when the model answered
    usefully, ``skipped`` when no call was made (disabled, or no credentials),
    ``failed`` when the call ran and didn't produce anything usable. The two
    non-ok states carry a ``reason``. ``call_log_entry`` is a cost-tracking
    dict in the pipeline's ``api_call_log`` shape when a call was attempted,
    else None.

    Never raises — a suggestion is a nicety, and no failure here may cost the
    author a review run.
    """
    seo_rules = (pub_config or {}).get("seo_rules") or {}
    if not seo_analysis.suggestions_enabled(seo_rules):
        return _skipped("disabled via seo_rules.suggestions")

    api_key = ((api_keys or {}).get("mistral") or {}).get("api_key", "")
    if not api_key:
        return _skipped("no mistral API key configured")

    meta_limit = seo_analysis.meta_description_limit(seo_rules)
    title_limit = seo_analysis.title_limit(seo_rules)

    seo_result = seo_result or {}
    title_too_long = any(
        issue.get("type") == "title_too_long" for issue in seo_result.get("issues", [])
    )
    og_title_request = ""
    if title_too_long:
        og_title_request = (
            f"OG TITLE REQUESTED: the article title is "
            f"{seo_result.get('title_length', 0)} characters, over this "
            f"publication's {title_limit}-character ceiling. Suggest an "
            f"og_title of at most {title_limit} characters that keeps the "
            f"meaning and reads as a title, not a truncation."
        )

    user_prompt = _build_user_prompt(
        text, handoff, pub_config, meta_limit, og_title_request
    )

    try:
        result = mistral.call(
            _SYSTEM_PROMPT,
            user_prompt,
            api_key,
            model=_SUGGESTION_MODEL,
        )
    except Exception as e:
        log.warning("SEO suggestion call raised: %s", e)
        return {"status": "failed", "reason": f"suggestion call raised: {e}"}, None

    call_log_entry = {
        "pass": CALL_LOG_PASS,
        "model": result.get("model", _SUGGESTION_MODEL),
        "failed": bool(result.get("failed")),
        "tokens": result.get("tokens", {}),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "error": result.get("error") if result.get("failed") else None,
    }

    if result.get("failed"):
        reason = f"suggestion call failed: {result.get('error', 'unknown error')}"
        log.warning("SEO suggestions unavailable — %s", reason)
        return {"status": "failed", "reason": reason}, call_log_entry

    data = result.get("data")
    if not isinstance(data, dict):
        reason = "suggestion call returned no usable JSON object"
        log.warning("SEO suggestions unavailable — %s", reason)
        return {"status": "failed", "reason": reason}, call_log_entry

    candidates = _clean_candidates(data.get("keyword_candidates"))
    meta, meta_chars, meta_over = _measured(data.get("meta_description"), meta_limit)

    if not candidates and not meta:
        reason = "suggestion call returned neither keyword candidates nor a description"
        log.warning("SEO suggestions unavailable — %s", reason)
        return {"status": "failed", "reason": reason}, call_log_entry

    suggestions = {
        "status": "ok",
        "model": call_log_entry["model"],
        "keyword_candidates": candidates,
        "meta_description": meta,
        "meta_description_chars": meta_chars,
        "meta_description_limit": meta_limit,
        "meta_description_over_limit": meta_over,
    }

    if title_too_long:
        og_title, og_chars, og_over = _measured(data.get("og_title"), title_limit)
        if og_title:
            suggestions.update(
                {
                    "og_title": og_title,
                    "og_title_chars": og_chars,
                    "og_title_limit": title_limit,
                    "og_title_over_limit": og_over,
                }
            )

    log.info(
        "SEO suggestions: %d keyword candidate(s), meta description %d chars%s",
        len(candidates),
        meta_chars,
        f" (OVER the {meta_limit}-char limit)" if meta_over else "",
    )
    return suggestions, call_log_entry
