"""Fetch a published web page and extract its main article content.

URL input mode for the pipeline: given a public URL, fetch the HTML (behind the
same SSRF guard used for link validation), strip boilerplate, and return a
synthesized handoff dict the review pipeline can consume.

Extraction prefers `trafilatura` (best-in-class main-content extraction) when it
is installed — it is an OPTIONAL dependency. When it is absent we fall back to a
built-in heuristic HTML parser that removes obvious boilerplate, prefers the
<article>/<main> region, and keeps h1-h3 headings (the SEO and structure checks
key off heading markup).
"""

import logging
import re
from html.parser import HTMLParser

import requests

from ci_core.http import USER_AGENT

from .links import _is_public_host

log = logging.getLogger(__name__)

_USER_AGENT = USER_AGENT
_FETCH_TIMEOUT = 20
# Below this, the page almost certainly didn't extract cleanly (paywall, JS-only
# render, or bot-block). We still run on what we got, but warn loudly.
_MIN_WORDS = 200

# Elements whose text is never article content.
_SKIP_TAGS = {
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "noscript",
    "form",
    "svg",
    "template",
}
# Block-level elements that mark a text-fragment boundary.
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "li",
    "tr",
    "br",
    "ul",
    "ol",
    "blockquote",
    "pre",
    "figure",
    "figcaption",
}
# Headings we preserve, mapped to their markdown prefix.
_HEADING_TAGS = {"h1": "#", "h2": "##", "h3": "###"}


class _ArticleParser(HTMLParser):
    """Heuristic main-content extractor used when trafilatura is unavailable.

    Collects text fragments tagged as paragraphs or headings, recording whether
    each fragment sits inside an <article>/<main> region so the caller can prefer
    that region. Boilerplate tags in ``_SKIP_TAGS`` are dropped wholesale.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_title = False
        self.title = ""
        self.main_depth = 0
        self.cur_heading = None
        self._buf = []
        self.nodes = []  # [{"kind": "#"|"##"|"###"|"p", "text": str, "in_main": bool}]

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        if not text:
            return
        self.nodes.append(
            {
                "kind": self.cur_heading or "p",
                "text": text,
                "in_main": self.main_depth > 0,
            }
        )

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
            return
        if tag in ("article", "main"):
            self.main_depth += 1
        if tag in _HEADING_TAGS:
            self._flush()
            self.cur_heading = _HEADING_TAGS[tag]
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
            return
        if tag in _HEADING_TAGS:
            self._flush()
            self.cur_heading = None
        elif tag in _BLOCK_TAGS:
            self._flush()
        if tag in ("article", "main") and self.main_depth:
            self.main_depth -= 1

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.in_title:
            self.title += data
            return
        if data.strip():
            self._buf.append(data)


def _fallback_body(nodes):
    """Render extracted nodes to markdown, preferring the <article>/<main> region."""
    chosen = [n for n in nodes if n["in_main"]] or nodes
    lines = []
    for n in chosen:
        if n["kind"] in ("#", "##", "###"):
            lines.append(f"{n['kind']} {n['text']}")
        else:
            lines.append(n["text"])
    return "\n\n".join(lines).strip()


def _extract_with_trafilatura(html):
    """Return markdown body via trafilatura, or None if unavailable/failed."""
    try:
        import trafilatura
    except ImportError:
        return None
    try:
        return trafilatura.extract(html, output_format="markdown", include_links=False)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(f"trafilatura extraction failed ({exc}); using built-in fallback.")
        return None


def extract_article(html):
    """Extract (title, body_markdown) from raw HTML.

    Title is the page <title> or, failing that, the first <h1>. Body comes from
    trafilatura when installed, otherwise the built-in heuristic parser.
    """
    parser = _ArticleParser()
    parser.feed(html)

    title = (parser.title or "").strip()
    if not title:
        for n in parser.nodes:
            if n["kind"] == "#":
                title = n["text"]
                break

    body = _extract_with_trafilatura(html)
    if not body:
        body = _fallback_body(parser.nodes)

    return (title or "Untitled"), (body or "").strip()


def fetch_url(url, timeout=_FETCH_TIMEOUT):
    """Fetch a public URL's HTML. Raises ValueError if the host is non-public.

    The SSRF guard runs BEFORE any network call — a user could paste any URL.
    """
    if not _is_public_host(url):
        raise ValueError(
            f"Refusing to fetch a non-public/internal host (SSRF guard): {url}"
        )
    resp = requests.get(
        url,
        timeout=timeout,
        allow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    )
    resp.raise_for_status()
    return resp.text


def build_handoff_from_url(url):
    """Fetch + extract a web page and synthesize an in-memory handoff dict.

    The pipeline only strictly needs ``title`` and ``draft``; the publication
    config still supplies voice_profile/audience to the review prompts. Optional
    handoff fields (primary_claim, pre_draft_analysis, etc.) are left unset —
    URL mode cannot infer an author's intent.
    """
    log.info(f"Fetching URL for review: {url}")
    html = fetch_url(url)
    title, draft = extract_article(html)
    draft = (draft or "").strip()

    word_count = len(draft.split())
    if word_count < _MIN_WORDS:
        log.warning(
            f"Extracted only {word_count} words from {url} — this is likely a paywall, "
            "a JavaScript-rendered page, or a bot-block. Running the review on the "
            "limited content that was extracted; results may be incomplete."
        )
    else:
        log.info(f"Extracted article '{title}' ({word_count} words).")

    return {"title": title or "Untitled", "draft": draft, "run_number": 1}
