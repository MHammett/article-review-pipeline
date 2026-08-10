"""Fetch a published web page and extract its main article content.

URL input mode for the pipeline: given a public URL, fetch the HTML (behind the
same SSRF guard used for link validation), strip boilerplate, and return a
synthesized handoff dict the review pipeline can consume.

The extraction itself lives in ``ci_core.extract`` — citation verification needs
the same boilerplate-stripping, and ``analysis`` cannot be imported from
``adapters`` without closing an import cycle. ``extract_article`` is re-exported
here so this module's callers keep working unchanged.
"""

import logging

import requests

from ci_core.extract import extract_article
from ci_core.http import USER_AGENT

from .links import _is_public_host

__all__ = ["extract_article", "fetch_url", "build_handoff_from_url"]

log = logging.getLogger(__name__)

_USER_AGENT = USER_AGENT
_FETCH_TIMEOUT = 20
# Below this, the page almost certainly didn't extract cleanly (paywall, JS-only
# render, or bot-block). We still run on what we got, but warn loudly.
_MIN_WORDS = 200


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
    config still supplies style_profile/audience to the review prompts. Optional
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
