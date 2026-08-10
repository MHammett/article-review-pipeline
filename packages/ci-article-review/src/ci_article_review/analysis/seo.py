"""Basic SEO structural analysis — no external APIs required."""

import logging
import re
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Defaults — overridden per-publication via the seo_rules: key in publication config.
_TITLE_MAX = 60
_TITLE_MIN = 20
_MIN_WORDS = 300
#: Search snippets truncate around here, and it's the limit publication.md's
#: SEO METADATA block already states to the author.
_META_DESCRIPTION_MAX = 155
#: Below this a description is too thin to be the snippet a searcher decides on;
#: search engines commonly replace short ones with body text of their choosing.
_META_DESCRIPTION_MIN = 70

#: Link text that tells a reader — and a search engine — nothing about the
#: destination. Compared casefolded against the full anchor text, so "here" is
#: flagged and "here is the county's filing" is not.
_WEAK_ANCHOR_TEXT = frozenset(
    {
        "click here",
        "here",
        "this",
        "this one",
        "this link",
        "this article",
        "this page",
        "link",
        "read more",
        "more",
        "learn more",
        "see more",
        "details",
        "full story",
    }
)

#: Inline markdown image: ``![alt](src)``. Alt text is group 1 and may be empty,
#: which is the case worth flagging.
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)")
#: Inline markdown link, excluding images via the negative lookbehind.
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)")
#: Raw HTML images, which reach the draft through embeds the pipeline preserves
#: verbatim. Same alt-text question applies to them.
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_HTML_ALT_RE = re.compile(r"""\balt\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
#: Anchor text that is itself a bare URL — reads as noise and describes nothing.
_BARE_URL_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)

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


def content_review_enabled(seo_rules=None):
    """Whether the SEO content review may make its model call. Defaults on.

    Separate from ``suggestions`` because it answers a different question and
    costs its own call: metadata suggestions fill in fields, this judges the
    article's structure for a search reader. ``--no-seo-suggestions`` turns
    both off; ``seo_rules.content_review: false`` turns off only this one.
    """
    return _bool_rule(seo_rules or {}, "content_review", True)


def _count_words(text):
    # Same pattern as readability.py so word counts are consistent across both modules.
    return len(re.findall(r"\b[a-zA-Z']+\b", text))


def _normalize_phrase(text):
    """Casefold and collapse whitespace, for comparing titles and headings."""
    return " ".join(str(text or "").split()).casefold()


def _image_alt_issues(text):
    """Images carrying no alt text.

    Alt text is the only description of an image available to a screen reader
    or an image-search crawler. One issue for the batch rather than one per
    image: a gallery post would otherwise bury every other finding.
    """
    missing = sum(1 for alt in _MD_IMAGE_RE.findall(text) if not alt[0].strip())
    for tag in _HTML_IMG_RE.findall(text):
        alt_match = _HTML_ALT_RE.search(tag)
        if not alt_match or not alt_match.group(1).strip():
            missing += 1

    if not missing:
        return []
    return [
        {
            "type": "missing_image_alt",
            "detail": (
                f"{missing} image(s) have no alt text — the only description a "
                "screen reader or an image-search crawler gets. Describe what "
                "the image shows, not what it is called."
            ),
        }
    ]


def _link_issues(text, site_url):
    """Anchor-text quality and internal linking.

    Both read the markdown link syntax, which ``analysis/links.py`` does not —
    that module regexes bare URLs to check whether they resolve, a different
    question from whether the link text says anything or points anywhere on
    your own site.
    """
    issues = []
    links = _MD_LINK_RE.findall(text)
    if not links:
        return issues

    weak = [
        anchor.strip()
        for anchor, _ in links
        if _normalize_phrase(anchor) in _WEAK_ANCHOR_TEXT
        or _BARE_URL_RE.match(anchor or "")
    ]
    if weak:
        sample = ", ".join(f'"{a}"' for a in sorted(set(weak))[:3])
        issues.append(
            {
                "type": "weak_anchor_text",
                "detail": (
                    f"{len(weak)} link(s) use text that describes nothing "
                    f"({sample}). Link text is a promise about the destination — "
                    "name what is on the other end."
                ),
            }
        )

    # Internal linking is only assessable when we know what "internal" means.
    host = urlparse(site_url).netloc.casefold().removeprefix("www.") if site_url else ""
    if not host:
        return issues

    def _is_internal(url):
        # A relative target is internal by construction; an absolute one has to
        # match the configured site host.
        if not url.startswith(("http://", "https://", "//", "mailto:")):
            return True
        return urlparse(url).netloc.casefold().removeprefix("www.") == host

    if not any(_is_internal(url) for _, url in links):
        issues.append(
            {
                "type": "no_internal_links",
                "detail": (
                    f"{len(links)} link(s), none of them to {host}. Linking your "
                    "own related coverage is how a reader (and a crawler) finds "
                    "the rest of it."
                ),
            }
        )
    return issues


def _title_h1_issues(title, h1s):
    """The handoff title and the article's H1 saying different things.

    Not an error — a search-facing title that differs from the on-page heading
    is a real tactic. But it is just as often an edit applied to one and not
    the other, and nothing else in the pipeline would notice.
    """
    if not title or not h1s:
        return []
    if _normalize_phrase(title) == _normalize_phrase(h1s[0]):
        return []
    return [
        {
            "type": "title_h1_mismatch",
            "detail": (
                f"Handoff title and article H1 differ — search shows "
                f"{title!r}, the page shows {h1s[0]!r}. Intentional is fine; "
                "confirm it is not an edit applied to only one of them."
            ),
        }
    ]


def _meta_description_length_issues(meta_description, max_chars, min_chars):
    """A supplied meta description outside the usable snippet range.

    Only presence was checked before, so an author who wrote a 300-character
    description got no warning that search would cut it mid-sentence.
    """
    length = len(meta_description.strip())
    if length > max_chars:
        return [
            {
                "type": "meta_description_too_long",
                "detail": (
                    f"{length} chars — over the {max_chars}-char limit, so "
                    "search will truncate it mid-sentence."
                ),
            }
        ]
    if length < min_chars:
        return [
            {
                "type": "meta_description_too_short",
                "detail": (
                    f"{length} chars — under {min_chars}, thin enough that "
                    "search engines often substitute body text instead."
                ),
            }
        ]
    return []


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


def keyword_usage(text, title, keyword):
    """Where ``keyword`` actually appears in the article.

    The point of proposing a focus keyword early is to catch the case where
    the article never uses the phrase it is supposed to rank for. Answering
    that is a substring scan, not a judgement call, so it costs nothing and
    runs over every candidate.

    Matching is casefolded and whitespace-collapsed but otherwise literal: no
    stemming, no synonyms. An article that says "interconnection queues" when
    the candidate is "interconnection queue" reports as absent, which
    overstates the problem slightly — but the alternative is a fuzzy match
    quietly reporting a phrase as present when a reader would not find it.
    """
    needle = _normalize_phrase(keyword)
    if not needle:
        return {
            "in_title": False,
            "in_headings": [],
            "in_opening": False,
            "body_count": 0,
        }

    headings = [h for _, h in re.findall(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE)]
    # "Opening" is the first paragraph after any leading H1 — the passage a
    # searcher reads before deciding to stay.
    body = re.sub(r"^#\s+.+$", "", text, count=1, flags=re.MULTILINE).strip()
    opening = body.split("\n\n", 1)[0] if body else ""

    return {
        "in_title": needle in _normalize_phrase(title),
        "in_headings": [h.strip() for h in headings if needle in _normalize_phrase(h)],
        "in_opening": needle in _normalize_phrase(opening),
        "body_count": _normalize_phrase(text).count(needle),
    }


def _annotate_keyword_usage(text, title, suggestions):
    """Attach a usage scan to each keyword candidate, in place."""
    for candidate in suggestions.get("keyword_candidates") or []:
        candidate["usage"] = keyword_usage(text, title, candidate.get("keyword", ""))


def apply_suggestions(seo_result, suggestions, text="", title=""):
    """Fold a generated suggestion block into an analysis result, in place.

    Kept out of ``analyze`` because ``analyze`` makes no API calls and the
    suggestions come from one (see ``analysis.seo_suggest``) — so a caller that
    doesn't want the call still gets a complete analysis.

    Beyond attaching the block, this does two things. Each keyword candidate
    gets a ``usage`` scan against ``text`` and ``title`` — free, and the whole
    reason for surfacing keywords at draft stage. And the
    ``no_meta_description`` issue is rewritten to point at the suggested
    description, which is why that issue survives in draft mode at all: paired
    with a concrete draft to edit it is a prompt, not a complaint.

    Returns ``seo_result`` for convenience.
    """
    if not suggestions:
        return seo_result

    seo_result["suggestions"] = suggestions
    if text:
        _annotate_keyword_usage(text, title or seo_result.get("title", ""), suggestions)
    # Read through the fields map rather than importing seo_suggest for an
    # accessor — that module imports this one, and the shape is one key deep.
    meta_field = (suggestions.get("fields") or {}).get("meta_description") or {}
    if not meta_field.get("value"):
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


def analyze(text, handoff=None, seo_rules=None, mode=DRAFT_MODE, site_url=None):
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
        meta_description_max_chars, meta_description_min_chars, suggestions,
        content_review. Falls back to module-level defaults when not provided.
    mode:
        Which handoff template this article arrived on — ``DRAFT_MODE`` or
        ``PUBLISH_MODE``. Only affects the wording of findings about metadata
        the draft template cannot carry; see ``_no_meta_description_detail``.
    site_url:
        The publication's own site URL (``wordpress.site_url``). Without it,
        internal linking can't be assessed and that check is skipped rather
        than guessed at.

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
    meta_max = meta_description_limit(seo_rules)
    meta_min = _positive_int(
        seo_rules, "meta_description_min_chars", _META_DESCRIPTION_MIN
    )

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
    else:
        # Presence was all this checked before, so a 300-character description
        # sailed through to be truncated mid-sentence in the search result.
        issues.extend(
            _meta_description_length_issues(
                seo_meta.get("meta_description")
                or seo_meta.get("seo_description")
                or "",
                meta_max,
                meta_min,
            )
        )

    # Flag H2/H3 imbalance: H3s without any H2 parent
    if h3s and not h2s:
        issues.append(
            {
                "type": "heading_hierarchy",
                "detail": f"{len(h3s)} H3 heading(s) present but no H2 — headings should nest logically",
            }
        )

    # On-page checks over the article's own markup: images, links, and the
    # title/H1 pair. Each returns at most one summarizing issue rather than one
    # per occurrence, so an image-heavy piece can't bury everything else.
    issues.extend(_image_alt_issues(text))
    issues.extend(_link_issues(text, site_url))
    issues.extend(_title_h1_issues(handoff.get("title", ""), h1s))

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
