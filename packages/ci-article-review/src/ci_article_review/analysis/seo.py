"""Basic SEO structural analysis — no external APIs required."""

import logging
import re

log = logging.getLogger(__name__)

# Defaults — overridden per-publication via the seo_rules: key in publication config.
_TITLE_MAX = 60
_TITLE_MIN = 20
_MIN_WORDS = 300


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


def _count_words(text):
    # Same pattern as readability.py so word counts are consistent across both modules.
    return len(re.findall(r"\b[a-zA-Z']+\b", text))


def analyze(text, handoff=None, seo_rules=None):
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
        Supported keys: title_max_chars, title_min_chars, min_article_words.
        Falls back to module-level defaults when not provided.

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
      issues          list  — [{type, detail}] for actionable problems
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
                "detail": "No meta description in handoff SEO METADATA section",
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
        "issues": issues,
    }
