"""Basic SEO structural analysis — no external APIs required."""

import logging
import re

log = logging.getLogger(__name__)

# Defaults — overridden per-publication via the seo_rules: key in publication config.
_TITLE_MAX = 60
_TITLE_MIN = 20
_MIN_WORDS = 300
#: Search snippets truncate around here, and it's the limit publication.md's
#: SEO METADATA block already states to the author.
_META_DESCRIPTION_MAX = 155

#: Which handoff template the article came in on. Draft submissions (Template A)
#: have no SEO section at all, so an absent SEO field means something different
#: there than it does in a publication handoff (Template C), which does.
DRAFT_MODE = "draft"
PUBLISH_MODE = "publish"


def _positive_int(seo_rules, key, default):
    """Read an int seo_rule, falling back to the default on a missing or bad value.

    A typo like ``title_max_chars: "sixty"`` in a publication config must not
    crash the pre-analysis pass — warn and use the default instead.
    """
    raw = seo_rules.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        log.warning(
            "seo_rules.%s = %r is not an integer — using default %d", key, raw, default
        )
        return default
    if value <= 0:
        log.warning(
            "seo_rules.%s = %r must be positive — using default %d", key, raw, default
        )
        return default
    return value


def _bool_rule(seo_rules, key, default):
    """Read a boolean seo_rule, falling back to the default on a bad value.

    Same contract as ``_positive_int``: a malformed publication config warns
    and degrades to the default rather than failing the run. Note that YAML
    already resolves ``no``/``off``/``false`` to a real bool, so a string here
    means the author quoted it or typed something else entirely.
    """
    raw = seo_rules.get(key, default)
    if isinstance(raw, bool):
        return raw
    log.warning(
        "seo_rules.%s = %r is not true/false — using default %r", key, raw, default
    )
    return default


def meta_description_limit(seo_rules=None):
    """Character ceiling for a meta description, from ``seo_rules`` or the default.

    Read by the suggestion pass (``analysis.seo_suggest``) so a generated draft
    description respects the same limit the publication handoff states.
    """
    return _positive_int(
        seo_rules or {}, "meta_description_max_chars", _META_DESCRIPTION_MAX
    )


def title_limit(seo_rules=None):
    """Character ceiling for the article title, from ``seo_rules`` or the default."""
    return _positive_int(seo_rules or {}, "title_max_chars", _TITLE_MAX)


def suggestions_enabled(seo_rules=None):
    """Whether the SEO suggestion pass may make its model call. Defaults on.

    On because it costs about the same as one citation-relevance check and it
    is what makes the "no meta description" finding actionable at draft stage;
    ``seo_rules.suggestions: false`` (or ``--no-seo-suggestions``) turns it off
    for anyone who doesn't want an extra call per run.
    """
    return _bool_rule(seo_rules or {}, "suggestions", True)


def _count_words(text):
    # Same pattern as readability.py so word counts are consistent across both modules.
    return len(re.findall(r"\b[a-zA-Z']+\b", text))


def _no_meta_description_detail(mode):
    """Wording for the ``no_meta_description`` issue, per handoff template.

    Template A (draft submission) has no SEO section at all, so in draft mode
    the old wording — "No meta description in handoff SEO METADATA section" —
    pointed the author at a section their document has no way to contain. It
    fired on every draft run and could never be cleared. Template C (the
    publication handoff) does have that section, so there the wording stands.

    Either way this stays an issue rather than being dropped: the field really
    is empty, and it really is required before publishing. What changes is that
    the draft-mode text says where it becomes due instead of implying the
    author omitted something. ``apply_suggestions`` then points it at a
    generated draft description when one is available.
    """
    if mode == PUBLISH_MODE:
        return "No meta description in handoff SEO METADATA section"
    return (
        "None supplied — the draft handoff (Template A) has no SEO section, so "
        "this is expected at draft stage. It becomes required in the publication "
        "handoff (Template C, SEO METADATA)."
    )


def apply_suggestions(seo_result, suggestions):
    """Fold a generated suggestion block into an analysis result, in place.

    Kept out of ``analyze`` because ``analyze`` makes no API calls and the
    suggestions come from one (see ``analysis.seo_suggest``) — so a caller that
    doesn't want the call still gets a complete analysis.

    Beyond attaching the block, this rewrites the ``no_meta_description`` issue
    to point at the suggested description. That is the whole reason the issue
    survives in draft mode: paired with a concrete draft to edit it is a
    prompt, not a complaint. Returns ``seo_result`` for convenience.
    """
    if not suggestions:
        return seo_result

    seo_result["suggestions"] = suggestions
    if not suggestions.get("meta_description"):
        return seo_result

    mode = seo_result.get("mode", DRAFT_MODE)
    for issue in seo_result.get("issues", []):
        if issue.get("type") != "no_meta_description":
            continue
        if mode == PUBLISH_MODE:
            issue["detail"] = (
                "No meta description in handoff SEO METADATA section — a "
                "suggested draft is below, under SEO suggestions."
            )
        else:
            issue["detail"] = (
                "None supplied — a suggested draft is below, under SEO "
                "suggestions. It belongs in the publication handoff "
                "(Template C, SEO METADATA)."
            )
    return seo_result


def analyze(text, handoff=None, seo_rules=None, mode=DRAFT_MODE):
    """Return an SEO analysis dict.

    Parameters
    ----------
    text:
        The article body (markdown).
    handoff:
        Parsed handoff dict (from handoff_parser). Used to read the article
        title and any SEO metadata the author supplied.
    seo_rules:
        Optional dict from the publication config's ``seo_rules:`` key.
        Supported keys: title_max_chars, title_min_chars, min_article_words,
        meta_description_max_chars, suggestions. Falls back to module-level
        defaults when not provided.
    mode:
        Which handoff template this article arrived on — ``DRAFT_MODE`` or
        ``PUBLISH_MODE``. Only affects the wording of findings about metadata
        the draft template cannot carry; see ``_no_meta_description_detail``.

    Returns
    -------
    dict with keys:
      title           str   — article title (from handoff or first H1)
      title_length    int
      h1_count        int
      h2_count        int
      h3_count        int
      word_count      int
      has_meta_description  bool
      mode            str   — the mode the findings were worded for
      issues          list  — [{type, detail}] for actionable problems

    ``apply_suggestions`` may later add a ``suggestions`` key.
    """
    handoff = handoff or {}
    seo_rules = seo_rules or {}
    title_max = _positive_int(seo_rules, "title_max_chars", _TITLE_MAX)
    title_min = _positive_int(seo_rules, "title_min_chars", _TITLE_MIN)
    min_words = _positive_int(seo_rules, "min_article_words", _MIN_WORDS)

    h1s = re.findall(r"^#\s+(.+)$", text, re.MULTILINE)
    h2s = re.findall(r"^##\s+(.+)$", text, re.MULTILINE)
    h3s = re.findall(r"^###\s+(.+)$", text, re.MULTILINE)

    title = handoff.get("title", "") or (h1s[0] if h1s else "")
    word_count = _count_words(text)

    seo_meta = handoff.get("seo", {}) or {}
    has_meta = bool(seo_meta.get("meta_description") or seo_meta.get("seo_description"))

    issues = []

    if title:
        if len(title) > title_max:
            issues.append(
                {
                    "type": "title_too_long",
                    "detail": f"{len(title)} chars — recommended ≤{title_max} for search snippets",
                }
            )
        elif len(title) < title_min:
            issues.append(
                {
                    "type": "title_too_short",
                    "detail": f"{len(title)} chars — recommended ≥{title_min}",
                }
            )
    else:
        issues.append(
            {"type": "no_title", "detail": "No title found in handoff or H1 heading"}
        )

    if len(h1s) == 0:
        issues.append({"type": "no_h1", "detail": "No H1 heading in article body"})
    elif len(h1s) > 1:
        issues.append(
            {
                "type": "multiple_h1",
                "detail": f"{len(h1s)} H1 headings found — should be exactly one",
            }
        )

    if word_count < min_words:
        issues.append(
            {
                "type": "thin_content",
                "detail": f"{word_count} words — recommended ≥{min_words} for indexability",
            }
        )

    if not has_meta:
        issues.append(
            {
                "type": "no_meta_description",
                "detail": _no_meta_description_detail(mode),
            }
        )

    # Flag H2/H3 imbalance: H3s without any H2 parent
    if h3s and not h2s:
        issues.append(
            {
                "type": "heading_hierarchy",
                "detail": f"{len(h3s)} H3 heading(s) present but no H2 — headings should nest logically",
            }
        )

    return {
        "title": title,
        "title_length": len(title),
        "h1_count": len(h1s),
        "h2_count": len(h2s),
        "h3_count": len(h3s),
        "word_count": word_count,
        "has_meta_description": has_meta,
        "mode": mode,
        "issues": issues,
    }
