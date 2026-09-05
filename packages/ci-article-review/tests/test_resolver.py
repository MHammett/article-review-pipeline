"""Tests for adapters.citation.resolver — parallel resolution, ordering, pointer flag."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from ci_core import extract
from ci_core.http import UnsafeURLError

from ci_article_review.adapters.citation import resolver, wayback


_SOURCES = [{"name": "FRED", "adapter": "fred"}]


def _no_wayback(url, timeout=10):
    return {"archived": None}


# A page whose <head>/nav boilerplate is bulky enough that head-of-raw-HTML
# truncation would show the verifier nothing but markup — the article body is
# what must reach the model.
_ARTICLE_HTML = (
    "<!DOCTYPE html><html><head><title>Example Report</title>"
    '<meta name="description" content="boilerplate">'
    "<script>var tracking = 1;</script></head>"
    "<body><nav>Home About Contact Subscribe Privacy</nav>"
    "<article><h1>Example Report</h1><p>"
    + ("The measured value is documented in detail throughout this report. " * 8)
    + "</p></article><footer>Copyright notice</footer></body></html>"
)


def _quote_from_prompt(user_prompt):
    """Return a sentence genuinely present in the prompt's page-content block.

    The relevance verifier requires a "supports" verdict to carry text copied
    from the page, and checks it (audit finding 1). Test doubles therefore have
    to behave like a model that actually read the document rather than one that
    asserts a verdict — deriving the quote from the prompt keeps them honest
    without hard-coding fixture text at every call site.
    """
    body = user_prompt
    if "<<<PAGE_CONTENT_" in body:
        body = body.split(">>>", 1)[-1].rsplit("<<<END_PAGE_CONTENT_", 1)[0]
    sentences = [s.strip() for s in body.split(".") if len(s.strip()) > 20]
    return (max(sentences, key=len) + ".") if sentences else body.strip()[:80]


def _page_response(body=_ARTICLE_HTML, content_type="text/html; charset=utf-8"):
    """Mock a ``requests`` response the resolver can actually extract from.

    The resolver reads ``.content``/``.headers``, not ``.text``, because it has
    to tell HTML from PDF and decode bytes itself.
    """
    return type(
        "R",
        (),
        {
            "raise_for_status": lambda self: None,
            "content": body.encode("utf-8") if isinstance(body, str) else body,
            "text": body if isinstance(body, str) else "",
            "headers": {"Content-Type": content_type},
            "encoding": "utf-8",
        },
    )()


def _prior_citation(url, content, verification="checksum", basis=None):
    """A section_9_citations entry as a prior run would have saved it.

    ``basis`` mirrors ``checksum_basis``, which drift comparison requires to
    match. It defaults to absent, which covers both adapter-sourced entries
    (their checksum basis never changed, so they carry no label) and reports
    written before ``known_url`` checksums moved from the raw response body to
    the extracted article text. Pass ``basis="extracted_text"`` for a prior
    standing in for a current-code ``known_url`` fetch.
    """
    entry = {
        "claim": "old claim",
        "url": url,
        "checksum": resolver.sha256_checksum(content),
        "resolved": True,
        "verification": verification,
    }
    if basis is not None:
        entry["checksum_basis"] = basis
    return entry


def _write_report(root, slug, run_number, ts, citations):
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"run_{run_number}_{ts.replace(':', '').replace('-', '')}_report.json"
    report = {
        "generated": ts,
        "run_number": run_number,
        "article_title": slug,
        "section_9_citations": citations,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return path


class TestResolveCitations:
    def test_empty_claims(self):
        assert resolver.resolve_citations([], _SOURCES) == []

    def test_preserves_claim_order(self):
        claims = ["claim zero", "claim one", "claim two"]

        def fake_resolve(claim, api_key=None):
            return {
                "found": True,
                "url": f"https://x/{claim.split()[-1]}",
                "content": claim,
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(claims, _SOURCES)

        assert [r["claim"] for r in results] == claims

    def test_unresolved_claim_marked(self):
        def fake_resolve(claim, api_key=None):
            return {"found": False}

        with patch(
            "ci_article_review.adapters.citation.sources.fred.resolve",
            side_effect=fake_resolve,
        ):
            results = resolver.resolve_citations(["unknown claim"], _SOURCES)

        assert results[0]["resolved"] is False
        assert "note" in results[0]

    def test_checksum_verification_label(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        assert results[0]["verification"] == "checksum"

    def test_pointer_only_label(self):
        def fake_resolve(claim, api_key=None):
            return {
                "found": True,
                "pointer_only": True,
                "url": "https://x",
                "content": "ptr",
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        assert results[0]["verification"] == "pointer"

    def test_missing_source_name_does_not_crash(self):
        sources = [{"adapter": "fred"}]  # no "name" key

        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(["c"], sources)

        # Falls back to adapter name instead of raising KeyError
        assert results[0]["source_name"] == "fred"

    def test_adapter_exception_is_isolated(self):
        def boom(claim, api_key=None):
            raise RuntimeError("adapter exploded")

        with patch(
            "ci_article_review.adapters.citation.sources.fred.resolve", side_effect=boom
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        # Exception is caught; claim reported unresolved rather than crashing the run
        assert results[0]["resolved"] is False

    def test_known_url_resolves_without_adapter_loop(self):
        """A claim carrying a known_url (e.g. supplied by the fact-check model)
        must be fetched and checksummed directly, never touching the adapter loop.
        """
        mock_resp = _page_response()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=mock_resp,
            ) as mock_get,
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve"
            ) as mock_adapter,
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is True
        assert results[0]["url"] == "https://example.com/page"
        assert results[0]["source_name"] == "fact-check model"
        # No mistral key is configured here, so relevance is never assessed and
        # the citation cannot claim checksum-level confidence.
        assert results[0]["verification"] == "unverifiable"
        mock_get.assert_called_once()
        mock_adapter.assert_not_called()

    def test_known_url_fetch_failure_reports_unresolved(self):
        with patch(
            "ci_article_review.adapters.citation.resolver.safe_get",
            side_effect=RuntimeError("timeout"),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/broken"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is False
        assert "note" in results[0]

    def test_no_api_keys_skips_relevance_check(self):
        """Without a mistral API key the relevance check cannot run, so the
        citation resolves (the URL fetched and checksummed fine) but is reported
        as unverifiable rather than as checksum-verified or as a mismatch."""
        mock_resp = _page_response()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider"
            ) as mock_mistral,
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
            )

        mock_mistral.assert_not_called()
        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "unverifiable"
        assert "relevance_check" in results[0]
        # Must never read as a finding against the source.
        assert "does not support" not in results[0]["note"]

    def test_relevance_check_supports_claim_stays_checksum_verified(self):
        mock_resp = _page_response()
        call_log = []

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                return_value={
                    "failed": False,
                    "data": {
                        "verdict": "supports",
                        "reason": "matches",
                        "quote": "The measured value is documented in detail throughout this report.",
                    },
                    "model": "mistral-small-latest",
                    "tokens": {"prompt": 10, "completion": 5},
                    "elapsed_seconds": 0.2,
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
                verification_call_log=call_log,
            )

        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "checksum"
        assert results[0]["relevance_verdict"] == "supports"
        assert len(call_log) == 1
        assert call_log[0]["failed"] is False
        assert call_log[0]["model"] == "mistral-small-latest"

    def test_relevance_check_contradicts_downgrades_citation(self):
        mock_resp = _page_response()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                return_value={
                    "failed": False,
                    "data": {
                        "verdict": "not_addressed",
                        "reason": "page never mentions this",
                    },
                    "model": "mistral-small-latest",
                    "tokens": {"prompt": 10, "completion": 5},
                    "elapsed_seconds": 0.2,
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
            )

        assert results[0]["resolved"] is False
        assert results[0]["verification"] != "checksum"
        assert "note" in results[0]

    def test_relevance_check_call_failure_degrades_gracefully(self):
        """The verification call itself failing (rate limit, timeout, etc.)
        must not crash resolution or block the citation — it degrades back
        to the pre-existing unverified-relevance behavior with a note."""
        mock_resp = _page_response()
        call_log = []

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                return_value={
                    "failed": True,
                    "error": "rate limited",
                    "model": "mistral-small-latest",
                    "tokens": {},
                    "elapsed_seconds": 0.1,
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
                verification_call_log=call_log,
            )

        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "unverifiable"
        assert "relevance_check" in results[0]
        assert "does not support" not in results[0]["note"]
        assert len(call_log) == 1
        assert call_log[0]["failed"] is True

    def test_relevance_check_exception_does_not_crash_resolution(self):
        mock_resp = _page_response()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                side_effect=RuntimeError("boom"),
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
            )

        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "unverifiable"
        assert "relevance_check" in results[0]

    def test_dict_entry_without_known_url_uses_adapter_loop(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": None}], _SOURCES
            )

        assert results[0]["resolved"] is True
        assert results[0]["source_name"] == "FRED"


class TestMultipleCitedSources:
    """A draft that cites several sources for a passage gets all of them checked.

    Reporting "the source does not support this" after checking one of three
    cited sources says something false about a draft that cited the right source
    second. Checking stops at the first that supports, so the extra candidates
    are only paid for by claims that would otherwise be reported unsupported.
    """

    def _verdicts(self, *by_url):
        """Return an llm.call_provider double that answers per fetched URL."""
        table = dict(by_url)
        seen = []

        def _fake_get(url, timeout=15):
            seen.append(url)
            return _page_response()

        def _fake_call(_provider, _system, user_prompt, _api_key, model=None):
            verdict = table[seen[-1]]
            return {
                "failed": False,
                "data": {
                    "verdict": verdict,
                    "reason": f"verdict for {seen[-1]}",
                    "quote": _quote_from_prompt(user_prompt),
                },
                "model": "mistral-small-latest",
                "tokens": {"prompt": 10, "completion": 5},
                "elapsed_seconds": 0.1,
            }

        return _fake_get, _fake_call, seen

    def _resolve(self, claim_entry, fake_get, fake_call):
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                side_effect=fake_get,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                side_effect=fake_call,
            ),
        ):
            return resolver.resolve_citations(
                [claim_entry], _SOURCES, api_keys={"mistral": {"api_key": "k"}}
            )[0]

    def test_checking_stops_at_the_first_source_that_supports(self):
        fake_get, fake_call, seen = self._verdicts(
            ("https://a.example", "supports"),
            ("https://b.example", "contradicts"),
        )
        result = self._resolve(
            {
                "claim": "a claim",
                "known_urls": ["https://a.example", "https://b.example"],
            },
            fake_get,
            fake_call,
        )
        assert result["verification"] == "checksum"
        assert result["url"] == "https://a.example"
        assert seen == ["https://a.example"]

    def test_a_later_cited_source_can_still_verify_the_claim(self):
        """The regression that motivated escalation.

        The first source cited for a passage often covers the sentence *after*
        the claim. Stopping there reports a false "does not support".
        """
        fake_get, fake_call, _ = self._verdicts(
            ("https://a.example", "not_addressed"),
            ("https://b.example", "supports"),
        )
        result = self._resolve(
            {
                "claim": "a claim",
                "known_urls": ["https://a.example", "https://b.example"],
            },
            fake_get,
            fake_call,
        )
        assert result["verification"] == "checksum"
        assert result["url"] == "https://b.example"
        assert result["alternates_checked"] == ["https://a.example"]

    def test_a_contradiction_outranks_a_not_addressed(self):
        """The finding that has to survive.

        A draft said "17 billion gallons"; the LBNL report it cited said 66
        billion liters (≈17.4 billion gallons) while two sibling sources simply
        did not discuss it. Reporting whichever came back first would bury the
        only thing worth acting on.
        """
        fake_get, fake_call, _ = self._verdicts(
            ("https://golf.example", "not_addressed"),
            ("https://usgs.example", "not_addressed"),
            ("https://lbnl.example", "contradicts"),
        )
        result = self._resolve(
            {
                "claim": "17 billion gallons",
                "known_urls": [
                    "https://golf.example",
                    "https://usgs.example",
                    "https://lbnl.example",
                ],
            },
            fake_get,
            fake_call,
        )
        assert result["verification"] == "content_mismatch"
        assert result["relevance_verdict"] == "contradicts"
        assert result["url"] == "https://lbnl.example"
        assert sorted(result["alternates_checked"]) == [
            "https://golf.example",
            "https://usgs.example",
        ]

    def test_the_note_says_the_other_cited_sources_were_checked_too(self):
        fake_get, fake_call, _ = self._verdicts(
            ("https://a.example", "not_addressed"),
            ("https://b.example", "not_addressed"),
        )
        result = self._resolve(
            {
                "claim": "a claim",
                "known_urls": ["https://a.example", "https://b.example"],
            },
            fake_get,
            fake_call,
        )
        assert "Also checked 1 other source(s)" in result["note"]

    def test_a_single_cited_source_costs_a_single_fetch(self):
        fake_get, fake_call, seen = self._verdicts(
            ("https://a.example", "not_addressed")
        )
        self._resolve(
            {"claim": "a claim", "known_urls": ["https://a.example"]},
            fake_get,
            fake_call,
        )
        assert seen == ["https://a.example"]

    def test_the_singular_known_url_key_still_works(self):
        """Kept so a caller (or a saved claim list) written against the old
        one-URL shape does not silently resolve nothing."""
        fake_get, fake_call, _ = self._verdicts(("https://a.example", "supports"))
        result = self._resolve(
            {"claim": "a claim", "known_url": "https://a.example"},
            fake_get,
            fake_call,
        )
        assert result["verification"] == "checksum"
        assert result["url"] == "https://a.example"


class TestVerifierSeesReadableContent:
    """Regression: the resolver used to hand `resp.text` — raw HTML source or
    raw PDF bytes — to the relevance verifier, which then rejected essentially
    every citation. The first 4000 characters of a real page are doctype, meta,
    script and nav tags; the article body never appeared in the window.
    """

    # Boilerplate long enough to fill the old head=4000 truncation window on its
    # own, and containing nothing about the claim.
    _BOILERPLATE = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        + "".join(
            f'<link rel="stylesheet" href="/assets/s{i}.css">'
            f'<script src="/assets/b{i}.js"></script>'
            for i in range(40)
        )
        + "<title>eGRID</title></head><body>"
        "<nav>Home Newsroom Regulations About Contact Subscribe</nav>"
    )
    _CLAIM = "eGRID subregions are EPA's emissions accounting zones"
    _BODY = (
        "<article><p>The Emissions &amp; Generation Resource Integrated Database "
        "(eGRID) is a comprehensive source of data on the environmental "
        "characteristics of almost all electric power generated in the United "
        "States. eGRID reports emissions accounting zones, known as eGRID "
        "subregions, which roughly follow the boundaries of the regional "
        "transmission organizations. The data includes emissions rates, net "
        "generation, and resource mix for each subregion.</p></article>"
    )

    def _run(self, html, verdict="supports"):
        captured = {}

        def fake_call(provider, system, user, api_key, model=None):
            captured["user"] = user
            return {
                "failed": False,
                # A real model quotes the page it was shown; the verifier now
                # checks that the quote is actually there (audit finding 1).
                "data": {
                    "verdict": verdict,
                    "reason": "r",
                    "quote": _quote_from_prompt(user),
                },
                "model": "mistral-small-latest",
                "tokens": {},
                "elapsed_seconds": 0.1,
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=_page_response(html),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                side_effect=fake_call,
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": self._CLAIM, "known_url": "https://www.epa.gov/egrid"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
            )
        return results[0], captured.get("user", "")

    def test_claim_supported_only_in_body_still_verifies(self):
        """The canonical smoke test in miniature: nav/boilerplate does NOT
        mention the claim, the article body DOES. Must verify."""
        result, prompt = self._run(self._BOILERPLATE + self._BODY + "</body></html>")

        assert result["verification"] == "checksum"
        assert result["resolved"] is True
        # What actually reached the model is the article text, not markup.
        assert "eGRID subregions" in prompt
        assert "stylesheet" not in prompt
        assert "<!DOCTYPE" not in prompt

    def test_checksum_covers_extracted_text_not_raw_html(self):
        result, _prompt = self._run(self._BOILERPLATE + self._BODY + "</body></html>")

        assert "<script" not in result["content_summary"]
        assert "eGRID" in result["content_summary"]

    def test_genuine_mismatch_still_downgrades(self):
        """The fix must not turn the verifier off — a real non-supporting
        verdict on readable text still downgrades the citation."""
        result, _prompt = self._run(
            self._BOILERPLATE + self._BODY + "</body></html>", verdict="not_addressed"
        )

        assert result["verification"] == "content_mismatch"
        assert result["resolved"] is False


class TestPdfCitations:
    """PDFs are primary sources here (ICNIRP, NIOSH, EPA rules). They must never
    be reported as content_mismatch on the strength of their raw bytes.
    """

    _CLAIM = "ICNIRP sets a 200 microtesla reference level at 50 Hz"
    _URL = "https://www.icnirp.org/cms/upload/publications/ICNIRPLFgdl.pdf"

    def _run(self, extracted_text):
        captured = {}

        def fake_call(provider, system, user, api_key, model=None):
            captured["user"] = user
            return {
                "failed": False,
                "data": {
                    "verdict": "supports",
                    "reason": "r",
                    "quote": _quote_from_prompt(user),
                },
                "model": "mistral-small-latest",
                "tokens": {},
                "elapsed_seconds": 0.1,
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=_page_response(
                    b"%PDF-1.6 binary body", content_type="application/pdf"
                ),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.extract"
                ".extract_response_text",
                return_value=(extracted_text, "pdf"),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                side_effect=fake_call,
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": self._CLAIM, "known_url": self._URL}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
            )
        return results[0], captured.get("user", "")

    def test_readable_pdf_verifies_on_its_text(self):
        result, prompt = self._run(
            "ICNIRP guidelines. " * 20
            + "The reference level for the general public is 200 microtesla at 50 Hz."
        )

        assert result["verification"] == "checksum"
        assert result["content_kind"] == "pdf"
        assert "200 microtesla" in prompt
        assert "%PDF" not in prompt

    def test_unreadable_pdf_is_unverifiable_not_a_mismatch(self):
        """A scanned PDF (or one we cannot parse) means we could not read the
        source — it does not mean the source fails to support the claim."""
        result, prompt = self._run("")

        assert result["verification"] == "unverifiable"
        assert result["verification"] != "content_mismatch"
        assert prompt == ""  # verification never ran
        assert "not be verified" in result["note"] or "NOT assessed" in result["note"]
        assert "does not support" not in result["note"]

    def test_unreadable_pdf_still_reports_the_source(self):
        """resolved stays True: we did fetch a real document, and it should
        still be archived and shown to the author for manual checking."""
        result, _prompt = self._run("")

        assert result["resolved"] is True
        assert result["url"] == self._URL


class TestAccessWallIsNotAMismatch:
    """eCFR (among others) serves a CAPTCHA interstitial with HTTP 200. It
    extracts into clean prose, so the verifier used to read the blocking notice
    and report that the cited regulation does not support the claim.
    """

    _WALL = (
        "<html><head><title>Request Access</title></head><body><main>"
        "<h1>Request Access</h1><p>Due to aggressive automated scraping of "
        "FederalRegister.gov and eCFR.gov, programmatic access to these sites is "
        "limited to our developer APIs. Your request has been flagged as "
        "potentially automated. If you are a human user receiving this message, "
        "please complete the CAPTCHA (bot test) below and click Request Access."
        "</p></main></body></html>"
    )

    def test_wall_reports_unverifiable_and_never_calls_the_verifier(self):
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=_page_response(self._WALL),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider"
            ) as mock_mistral,
        ):
            results = resolver.resolve_citations(
                [
                    {
                        "claim": "Backup generators may run 50-100 hours per year.",
                        "known_url": "https://www.ecfr.gov/current/title-40",
                    }
                ],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
            )

        assert results[0]["verification"] == "unverifiable"
        assert results[0]["content_kind"] == "access_wall"
        assert "does not support" not in results[0]["note"]
        mock_mistral.assert_not_called()


def _http_error_response(status_code):
    resp = MagicMock()
    resp.status_code = status_code
    error = requests.exceptions.HTTPError(f"{status_code} error", response=resp)
    resp.raise_for_status.side_effect = error
    return resp


class TestKnownUrlWaybackFallback:
    """A known_url the origin refused (401/403/429) or that we never reached
    (timeout, DNS/connection error) should fall back to a Wayback snapshot;
    404/410/5xx should not — see resolver._wayback_fallback_content scoping."""

    def test_403_recovers_via_wayback_snapshot(self):
        snapshot_url = (
            "https://web.archive.org/web/20240101000000/https://example.com/page"
        )
        # The archived copy is HTML too (with archive.org's banner on top), so
        # it goes through the same article extraction as a direct fetch.
        snap_resp = _page_response()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                side_effect=[_http_error_response(403), snap_resp],
            ) as mock_get,
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                return_value={"archived": True, "snapshot_url": snapshot_url},
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is True
        assert results[0]["verified_via"] == "wayback_fallback"
        # Checksum covers the extracted article text, not the raw snapshot HTML.
        expected_text, _ = extract.extract_response_text(
            _ARTICLE_HTML.encode("utf-8"), content_type="text/html"
        )
        assert results[0]["checksum"] == resolver.sha256_checksum(expected_text)
        assert "<html" not in results[0]["content_summary"]
        assert mock_get.call_count == 2

    def test_403_with_no_snapshot_reports_unresolved(self):
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=_http_error_response(403),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                return_value={"archived": False},
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/blocked"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is False
        assert "note" in results[0]

    def test_404_does_not_attempt_wayback_fallback(self):
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=_http_error_response(404),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check"
            ) as mock_wb,
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/gone"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is False
        mock_wb.assert_not_called()

    def _resolve_with_fetch_failure(self, exc, wayback_result):
        """Run one known_url claim whose direct fetch raises ``exc``.

        ``wayback.submit`` is stubbed alongside ``check``. It has to be: a
        resolved result whose snapshot is absent or stale is a re-capture target
        for ``_submit_missing_archives``, and ``example.com`` passes
        ``is_public_host``, so an unstubbed run asks archive.org's Save Page Now
        API to really capture the page — measured at 21s of live network in a
        unit test, on every run of the suite.
        """
        snap_resp = _page_response()
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                side_effect=[exc, snap_resp],
            ) as mock_get,
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                return_value=wayback_result,
            ) as mock_wb,
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
                return_value={"submitted": True, "job_id": None},
            ) as mock_submit,
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
            )
        return results[0], mock_get, mock_wb, mock_submit

    def test_timeout_recovers_via_wayback_snapshot(self):
        snapshot_url = (
            "https://web.archive.org/web/20240101000000/https://example.com/page"
        )
        result, mock_get, _, _ = self._resolve_with_fetch_failure(
            requests.exceptions.ReadTimeout("read timed out"),
            {"archived": True, "snapshot_url": snapshot_url},
        )

        assert result["resolved"] is True
        assert result["verified_via"] == "wayback_fallback"
        assert result["origin_failure"] == "timeout"
        assert "archive.org snapshot" in result["archive_provenance"]
        assert mock_get.call_count == 2

    def test_dns_failure_recovers_via_wayback_snapshot(self):
        snapshot_url = (
            "https://web.archive.org/web/20240101000000/https://example.com/page"
        )
        # requests wraps urllib3's NameResolutionError in a ConnectionError.
        result, _, _, _ = self._resolve_with_fetch_failure(
            requests.exceptions.ConnectionError(
                "NameResolutionError: getaddrinfo failed [Errno 11002]"
            ),
            {"archived": True, "snapshot_url": snapshot_url},
        )

        assert result["resolved"] is True
        assert result["verified_via"] == "wayback_fallback"
        assert result["origin_failure"] == "unreachable"

    def test_timeout_with_no_snapshot_reports_unresolved(self):
        result, _, _, _ = self._resolve_with_fetch_failure(
            requests.exceptions.Timeout("timed out"), {"archived": False}
        )

        assert result["resolved"] is False
        assert "could not be fetched" in result["note"]
        assert "verified_via" not in result

    def test_stale_snapshot_stays_flagged_stale(self):
        """A 245-day-old snapshot satisfying a timeout is still stale."""
        result, _, mock_wb, mock_submit = self._resolve_with_fetch_failure(
            requests.exceptions.Timeout("timed out"),
            {
                "archived": True,
                "snapshot_url": "https://web.archive.org/web/20240101000000/https://example.com/page",
                "snapshot_age_days": 245,
                "snapshot_stale": True,
            },
        )

        assert result["wayback"]["snapshot_stale"] is True
        assert result["wayback"]["snapshot_age_days"] == 245
        assert "245 days old" in result["archive_provenance"]
        # The availability answer from the fallback is reused, not re-fetched.
        assert mock_wb.call_count == 1
        # ...and a stale snapshot is queued for re-capture, which is the other
        # half of _submit_missing_archives' contract. This ran unasserted
        # against the live Save Page Now API before the stub above.
        assert mock_submit.call_count == 1
        assert mock_submit.call_args.args[0] == "https://example.com/page"

    def test_5xx_does_not_attempt_wayback_fallback(self):
        """An origin-side error is the origin's problem, not something an
        archive copy should paper over — see wayback._FALLBACK_STATUSES."""
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=_http_error_response(503),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check"
            ) as mock_wb,
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/down"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is False
        mock_wb.assert_not_called()

    def test_429_recovers_via_wayback_snapshot(self):
        snapshot_url = (
            "https://web.archive.org/web/20240101000000/https://example.com/page"
        )
        snap_resp = _page_response()
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                side_effect=[_http_error_response(429), snap_resp],
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                return_value={"archived": True, "snapshot_url": snapshot_url},
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
            )

        assert results[0]["resolved"] is True
        assert results[0]["origin_failure"] == "rate_limited"


class TestUnresolvedCitationsRecordWhatArchiveOrgSaid:
    """A failed fetch used to record no archive state at all.

    ``_wayback_fallback_content`` asked archive.org, got an answer that did not
    yield readable content, and dropped it on the way out. The caller's
    ``resolved: False`` citation therefore said neither "archive.org has no
    snapshot" nor "the lookup never completed" — and the rate limiter's circuit
    breaker makes the second the common case in a throttled run rather than a
    rare one, so a reader had no way to tell which had happened.

    The absence of the key is itself meaningful and has to stay that way: a
    failure that never qualified for the fallback never asked anyone.
    """

    URL = "https://example.com/page"

    def _resolve(self, origin, wayback_result=None, snapshot=None):
        """Resolve one known_url claim whose direct fetch fails.

        ``origin`` is what the first ``safe_get`` does — an exception instance
        to raise, or a response whose ``raise_for_status`` raises. ``snapshot``
        is what the *second* call does, for the case where a snapshot exists but
        cannot be read; omit it when no snapshot fetch is expected.
        """
        fetches = [origin] if snapshot is None else [origin, snapshot]
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                side_effect=fetches,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                return_value=wayback_result,
            ) as mock_wb,
        ):
            result = resolver._resolve_known_url("a claim", self.URL)
        return result, mock_wb

    def test_no_snapshot_is_recorded_as_no_snapshot(self):
        """archive.org answered "nothing archived". That is a fact worth keeping."""
        result, mock_wb = self._resolve(
            requests.exceptions.Timeout("timed out"),
            {"url": self.URL, "archived": False},
        )
        assert result["resolved"] is False
        assert result["wayback"]["archived"] is False
        assert mock_wb.call_count == 1

    def test_a_lookup_that_never_completed_is_recorded_as_such(self):
        """The breaker's null must not read as "no snapshot" — it is "we never
        found out", which is what a throttled run is full of."""
        result, _ = self._resolve(
            _http_error_response(403),
            {
                "url": self.URL,
                "archived": None,
                "error": (
                    "skipped: archive.org rate limit tripped earlier this run "
                    "(no snapshot lookup attempted)"
                ),
            },
        )
        assert result["resolved"] is False
        assert result["wayback"]["archived"] is None
        assert "rate limit" in result["wayback"]["error"]

    def test_a_stale_snapshot_survives_a_failed_snapshot_fetch(self):
        """archive.org has a snapshot; we just could not read it this run.

        The snapshot URL and its staleness are the most useful thing the run
        learned about this citation — the author can open the copy by hand — and
        they were exactly what the old contract threw away.
        """
        snapshot_url = (
            "https://web.archive.org/web/20240101000000/https://example.com/page"
        )
        result, _ = self._resolve(
            requests.exceptions.Timeout("timed out"),
            {
                "url": self.URL,
                "archived": True,
                "snapshot_url": snapshot_url,
                "snapshot_age_days": 245,
                "snapshot_stale": True,
            },
            snapshot=requests.exceptions.ConnectionError("snapshot unreachable"),
        )
        assert result["resolved"] is False
        assert result["wayback"]["snapshot_url"] == snapshot_url
        assert result["wayback"]["snapshot_stale"] is True
        assert result["wayback"]["snapshot_age_days"] == 245
        # Read from the archive is what `verified_via` claims; nothing was read.
        assert "verified_via" not in result

    def test_a_404_carries_no_wayback_key_because_nobody_was_asked(self):
        """No lookup happened, so there is no answer to record. An empty dict or
        a null here would both say a lookup ran, which is the confusion this
        whole change exists to remove."""
        result, mock_wb = self._resolve(_http_error_response(404))

        assert result["resolved"] is False
        mock_wb.assert_not_called()
        assert "wayback" not in result

    def test_a_refused_private_address_carries_no_wayback_key(self):
        """The other route to "never asked": the SSRF guard refuses before any
        fetch, and asking archive.org would hand the internal URL to a third
        party. See _resolve_known_url's UnsafeURLError branch."""
        result, mock_wb = self._resolve(UnsafeURLError("blocked"))

        assert result["resolved"] is False
        mock_wb.assert_not_called()
        assert "wayback" not in result


class TestArchiveSubmission:
    @pytest.fixture(autouse=True)
    def _treat_fixture_urls_as_public(self):
        """Let the placeholder URLs in this class past the public-host check.

        Submission now skips non-public URLs so an internal hostname is never
        handed to archive.org (audit finding 20). These tests use unresolvable
        placeholders like ``https://x``, which the guard correctly rejects.
        Patching it here keeps the suite offline — the alternative is real DNS
        in unit tests — while the guard's own behaviour is covered in
        ``TestArchiveSubmissionSkipsNonPublicUrls`` below.
        """
        with patch(
            "ci_article_review.adapters.citation.resolver.is_public_host",
            return_value=True,
        ):
            yield

    def _archived_wayback(self, url, timeout=10):
        return {"archived": True}

    def _unarchived_wayback(self, url, timeout=10):
        return {"archived": False}

    def test_submits_only_when_not_archived(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=self._unarchived_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
                return_value={"submitted": True, "job_id": None},
            ) as mock_submit,
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        mock_submit.assert_called_once()
        assert mock_submit.call_args.args[0] == "https://x"
        assert results[0]["wayback"]["submitted"] is True

    def test_does_not_submit_when_already_archived(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=self._archived_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
            ) as mock_submit,
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        mock_submit.assert_not_called()
        assert "submitted" not in results[0]["wayback"]

    def test_submission_failure_does_not_raise_or_fail_resolution(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=self._unarchived_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
                side_effect=RuntimeError("rate limited"),
            ),
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        assert results[0]["resolved"] is True
        assert results[0]["wayback"]["submitted"] is False
        assert "rate limited" in results[0]["wayback"]["submission_error"]

    def test_archive_org_credentials_passed_through(self):
        def fake_resolve(claim, api_key=None):
            return {"found": True, "url": "https://x", "content": "data"}

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=self._unarchived_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
                return_value={"submitted": True, "job_id": "j1"},
            ) as mock_submit,
        ):
            resolver.resolve_citations(
                ["c"],
                _SOURCES,
                api_keys={"archive_org": {"access_key": "AK", "secret_key": "SK"}},
            )

        assert mock_submit.call_args.kwargs["access_key"] == "AK"
        assert mock_submit.call_args.kwargs["secret_key"] == "SK"

    def test_no_submission_when_unresolved(self):
        def fake_resolve(claim, api_key=None):
            return {"found": False}

        with (
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=fake_resolve,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.submit",
            ) as mock_submit,
        ):
            resolver.resolve_citations(["c"], _SOURCES)

        mock_submit.assert_not_called()


def _fred_returning(url, content, pointer_only=False):
    def fake_resolve(claim, api_key=None):
        result = {"found": True, "url": url, "content": content}
        if pointer_only:
            result["pointer_only"] = True
        return result

    return fake_resolve


def _resolve_against_history(history_root, fake_resolve, claims=None):
    with (
        patch(
            "ci_article_review.adapters.citation.resolver.wayback.check",
            side_effect=_no_wayback,
        ),
        patch(
            "ci_article_review.adapters.citation.sources.fred.resolve",
            side_effect=fake_resolve,
        ),
    ):
        return resolver.resolve_citations(
            claims or ["new claim"], _SOURCES, history_root=str(history_root)
        )


class TestContentDriftDetection:
    """Cross-run checksum comparison: a URL resolved at the checksum tier
    before, with a different checksum now, is flagged content_changed_since."""

    def test_same_url_same_checksum_no_drift(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "same content")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/data", "same content")
        )

        assert results[0]["verification"] == "checksum"
        assert "content_changed_since" not in results[0]

    def test_same_url_different_checksum_flags_drift(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            3,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "old content")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/data", "new content")
        )

        drift = results[0]["content_changed_since"]
        assert drift["prior_run"] == 3
        assert drift["prior_article"] == "article-one"
        assert drift["prior_date"] == "2026-01-01T00:00:00"
        assert drift["prior_checksum"] == resolver.sha256_checksum("old content")

    def test_drift_never_blocks_resolution(self, tmp_path):
        """A changed source is a reviewer signal, not a resolution failure."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "old content")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/data", "new content")
        )

        assert results[0]["resolved"] is True
        assert results[0]["verification"] == "checksum"
        assert "content_changed_since" in results[0]

    def test_never_before_seen_url_no_comparison(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/unrelated", "whatever")],
        )

        results = _resolve_against_history(
            tmp_path,
            _fred_returning("https://example.com/brand-new", "brand new content"),
        )

        assert results[0]["resolved"] is True
        assert "content_changed_since" not in results[0]

    def test_drift_detected_across_different_articles(self, tmp_path):
        """The same source URL cited by a different article's prior run still
        counts — sources get reused across articles for the same publication."""
        _write_report(
            tmp_path,
            "some-other-article",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/shared-source", "v1")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/shared-source", "v2")
        )

        assert results[0]["content_changed_since"]["prior_article"] == (
            "some-other-article"
        )

    def test_most_recent_prior_run_wins(self, tmp_path):
        """With several prior sightings of a URL, the comparison is against the
        newest one, not whichever file happened to be scanned last."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "v1")],
        )
        _write_report(
            tmp_path,
            "article-two",
            2,
            "2026-06-01T00:00:00",
            [_prior_citation("https://example.com/data", "v2")],
        )

        results = _resolve_against_history(
            tmp_path, _fred_returning("https://example.com/data", "v3")
        )

        drift = results[0]["content_changed_since"]
        assert drift["prior_article"] == "article-two"
        assert drift["prior_checksum"] == resolver.sha256_checksum("v2")

    def test_missing_history_root_no_crash(self, tmp_path):
        results = _resolve_against_history(
            tmp_path / "does-not-exist", _fred_returning("https://x", "data")
        )

        assert results[0]["resolved"] is True
        assert "content_changed_since" not in results[0]

    def test_no_history_root_skips_history_scan_entirely(self):
        """Callers that don't opt in pay nothing — no history scan happens."""
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.sources.fred.resolve",
                side_effect=_fred_returning("https://x", "data"),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver."
                "history_analytics.load_reports"
            ) as mock_load,
        ):
            results = resolver.resolve_citations(["c"], _SOURCES)

        mock_load.assert_not_called()
        assert "content_changed_since" not in results[0]

    def test_known_url_citation_gets_drift_check(self, tmp_path):
        """Drift applies to the known_url path too, not just adapter results."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [
                _prior_citation(
                    "https://example.com/page",
                    "old page content",
                    basis="extracted_text",
                )
            ],
        )
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=_page_response(),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                return_value={
                    "failed": False,
                    "data": {
                        "verdict": "supports",
                        "reason": "yes",
                        "quote": "The measured value is documented in detail throughout this report.",
                    },
                    "model": "mistral-small-latest",
                    "tokens": {},
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
                history_root=str(tmp_path),
            )

        assert results[0]["verification"] == "checksum"
        assert results[0]["content_changed_since"]["prior_checksum"] == (
            resolver.sha256_checksum("old page content")
        )

    def test_legacy_raw_html_checksum_does_not_report_false_drift(self, tmp_path):
        """Reports written before the checksum moved from the raw response body
        to the extracted article text carry no ``checksum_basis``. Comparing
        across that change would flag every previously-cited source as changed
        on the first run after the switch — a page that never moved reported as
        having moved.
        """
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [
                _prior_citation(
                    "https://example.com/page", "old raw html body", basis=None
                )
            ],
        )

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=_page_response(),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                return_value={
                    "failed": False,
                    "data": {
                        "verdict": "supports",
                        "reason": "yes",
                        "quote": "The measured value is documented in detail throughout this report.",
                    },
                    "model": "mistral-small-latest",
                    "tokens": {},
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
                history_root=str(tmp_path),
            )

        assert results[0]["verification"] == "checksum"
        assert results[0]["checksum_basis"] == "extracted_text"
        assert "content_changed_since" not in results[0]

    def test_content_mismatch_citation_not_drift_flagged(self, tmp_path):
        """A citation demoted to content_mismatch has left the checksum tier —
        it already carries a stronger signal, so drift doesn't pile on."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [
                _prior_citation(
                    "https://example.com/page",
                    "old page content",
                    basis="extracted_text",
                )
            ],
        )
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=_page_response(),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.llm.call_provider",
                return_value={
                    "data": {"verdict": "contradicts", "reason": "no"},
                    "model": "mistral-small-latest",
                    "tokens": {},
                },
            ),
        ):
            results = resolver.resolve_citations(
                [{"claim": "a claim", "known_url": "https://example.com/page"}],
                _SOURCES,
                api_keys={"mistral": {"api_key": "k"}},
                history_root=str(tmp_path),
            )

        assert results[0]["verification"] == "content_mismatch"
        assert "content_changed_since" not in results[0]


class TestChecksumIndexTierFiltering:
    """Only checksum-tier prior entries are indexed, and only checksum-tier
    resolutions are compared — a pointer-only checksum is taken over whatever
    the adapter called content (often nothing), so comparing it is noise."""

    def test_pointer_only_resolution_is_never_drift_flagged(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/portal", "old blurb")],
        )

        results = _resolve_against_history(
            tmp_path,
            _fred_returning(
                "https://example.com/portal", "new blurb", pointer_only=True
            ),
        )

        assert results[0]["verification"] == "pointer"
        assert "content_changed_since" not in results[0]

    def test_pointer_tier_prior_entry_is_not_indexed(self, tmp_path):
        """A URL last seen pointer-only must not make a later real fetch of
        the same URL look like the page changed."""
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/portal", "", verification="pointer")],
        )

        assert resolver.build_checksum_index(str(tmp_path)) == {}

    def test_prior_entry_without_verification_field_is_not_indexed(self, tmp_path):
        """Reports predating the confidence tiers can't be placed in a tier,
        so they're skipped rather than compared on a guess."""
        entry = _prior_citation("https://example.com/data", "content")
        del entry["verification"]
        _write_report(tmp_path, "article-one", 1, "2026-01-01T00:00:00", [entry])

        assert resolver.build_checksum_index(str(tmp_path)) == {}

    def test_unresolved_prior_entry_has_no_checksum_to_index(self, tmp_path):
        _write_report(
            tmp_path,
            "article-one",
            1,
            "2026-01-01T00:00:00",
            [
                {
                    "claim": "old claim",
                    "url": "https://example.com/data",
                    "resolved": False,
                    "note": "unfetchable",
                }
            ],
        )

        assert resolver.build_checksum_index(str(tmp_path)) == {}

    def test_checksum_tier_prior_entry_is_indexed(self, tmp_path):
        """Guard against the filters above passing vacuously — a well-formed
        checksum-tier entry must actually land in the index."""
        _write_report(
            tmp_path,
            "article-one",
            7,
            "2026-01-01T00:00:00",
            [_prior_citation("https://example.com/data", "content")],
        )

        index = resolver.build_checksum_index(str(tmp_path))

        assert index["https://example.com/data"] == {
            "checksum": resolver.sha256_checksum("content"),
            "checksum_basis": None,
            "article_slug": "article-one",
            "run_number": 7,
            "generated": "2026-01-01T00:00:00",
        }


class TestStaleSnapshotsAreResubmitted:
    """Staleness was detected, reported, and then ignored.

    The point of archiving a citation is that the page is still readable later.
    A snapshot older than the staleness threshold predates whatever the page
    says now, so it needs re-capturing just as much as a missing one does.
    """

    def _entry(self, **wayback):
        return {
            "resolved": True,
            "url": "https://example.org/report",
            "wayback": wayback,
        }

    def test_a_stale_snapshot_is_submitted(self):
        entries = [self._entry(archived=True, snapshot_stale=True)]
        with patch.object(
            resolver.wayback, "submit", return_value={"submitted": True}
        ) as mock_submit:
            resolver._submit_missing_archives(entries, {})
        mock_submit.assert_called_once()

    def test_a_fresh_snapshot_is_left_alone(self):
        entries = [self._entry(archived=True, snapshot_stale=False)]
        with patch.object(resolver.wayback, "submit") as mock_submit:
            resolver._submit_missing_archives(entries, {})
        mock_submit.assert_not_called()

    def test_an_absent_snapshot_is_still_submitted(self):
        entries = [self._entry(archived=False)]
        with patch.object(
            resolver.wayback, "submit", return_value={"submitted": True}
        ) as mock_submit:
            resolver._submit_missing_archives(entries, {})
        mock_submit.assert_called_once()

    def test_a_private_host_is_never_sent_even_when_stale(self):
        """Submitting would transmit an internal hostname to a third party."""
        entries = [
            {
                "resolved": True,
                "url": "http://10.1.8.25/internal/report",
                "wayback": {"archived": True, "snapshot_stale": True},
            }
        ]
        with patch.object(resolver.wayback, "submit") as mock_submit:
            resolver._submit_missing_archives(entries, {})
        mock_submit.assert_not_called()


class TestWaybackRateLimitHandling:
    """archive.org throttled every lookup and nothing backed off.

    Measured 2026-08-12: 12 consecutive availability requests all 429, the first
    included, still 429 after a 45s cooldown at 1 req/6s. The 2026-08-11 run
    archived 0 of 49 resolved citations. After pacing + backoff, 6 of 6 live
    lookups succeeded in 19.8s.
    """

    def setup_method(self):
        wayback.reset_rate_limit_state()

    def teardown_method(self):
        wayback.reset_rate_limit_state()

    def test_calls_are_paced_apart(self):
        # The interval is patched in explicitly rather than taken from the
        # module: conftest's `neutralise_wayback_pacing` zeroes it for every
        # test, so anything asserting the pacing has to ask for it back.
        # (`test_wayback.py::TestPacingClock` asserts the exact durations and
        # the interaction with a 429's backoff clock; this stays here as the
        # resolver-side statement that pacing happens at all.)
        seen = []
        with (
            patch.object(wayback, "_MIN_INTERVAL_SECONDS", 3.0),
            patch.object(wayback.time, "sleep", side_effect=seen.append),
        ):
            wayback._pace()
            wayback._pace()
        assert seen and seen[-1] > 0, "second call must wait for the interval"

    def test_a_429_is_retried_with_backoff(self):
        resp = MagicMock(status_code=429, headers={}, url="https://archive.org/x")
        with patch.object(wayback.requests, "get", return_value=resp) as mock_get:
            with patch.object(wayback.time, "sleep"):
                with pytest.raises(Exception):
                    wayback._get_availability("https://example.org", 10)
        assert mock_get.call_count == wayback._MAX_ATTEMPTS

    def test_retry_after_header_is_honoured(self):
        resp = MagicMock(status_code=429, headers={"Retry-After": "12"})
        assert wayback._retry_after_seconds(resp, 0) == 12.0

    def test_retry_after_is_capped(self):
        """A hostile or absurd header must not stall the whole run."""
        resp = MagicMock(status_code=429, headers={"Retry-After": "86400"})
        assert wayback._retry_after_seconds(resp, 0) == 60.0

    def test_the_circuit_breaker_stops_further_lookups(self):
        wayback._rate_limited_lookups = wayback._CIRCUIT_TRIP_AFTER
        with patch.object(wayback.requests, "get") as mock_get:
            result = wayback.check("https://example.org/page")
        mock_get.assert_not_called()
        assert result["archived"] is None
        assert "rate limit tripped" in result["error"]

    def test_a_success_does_not_erase_other_lookups_refusals(self):
        """The budget only moves up, and that is deliberate.

        This used to reset to zero on any success. Under the resolver's thread
        pool that meant one worker's 200 wiped out every other worker's
        refusals, so a run being throttled four-in-five never tripped the
        breaker — the exact situation it was built for. "Consecutive" is not a
        quantity eight interleaved threads can agree on; a per-run budget is.
        """
        wayback._rate_limited_lookups = 3
        ok = MagicMock(status_code=200, headers={})
        ok.raise_for_status = MagicMock()
        with patch.object(wayback.requests, "get", return_value=ok):
            with patch.object(wayback.time, "sleep"):
                wayback._get_availability("https://example.org", 10)
        assert wayback._rate_limited_lookups == 3


class TestResolvedUrlIsRecorded:
    """Where the fetch actually landed, for citations that arrive via a redirector.

    A grounded model cites through one: a gemini-sourced citation arrives as a
    271-character `vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIY...`
    that names no publication and is not durable. `safe_get` follows redirects a
    validated hop at a time and returns the last hop's response, so the resolved
    address was already in hand and was being dropped. Measured 2026-09-05 on the
    Honda run: opaque redirect URLs were 817 of the 2,682 characters in one
    Section 9 entry, while the page behind them was a Jalopnik article this pass
    had already fetched, read and checksummed.
    """

    _REDIRECTOR = "https://redirector.example/grounding-api-redirect/AUZIYabc123"
    _REAL = "https://www.jalopnik.com/honda-clocks-stuck"

    def _fetch(self, final_url):
        resp = MagicMock()
        resp.url = final_url
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.text = (
            "<html><body><article><p>" + ("word " * 80) + "</p></article></body></html>"
        )
        resp.content = resp.text.encode()
        resp.encoding = "utf-8"
        resp.apparent_encoding = "utf-8"
        resp.raise_for_status = MagicMock()
        return resp

    def _resolve(self, final_url):
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                return_value=self._fetch(final_url),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                return_value={"archived": False},
            ),
            patch(
                "ci_article_review.adapters.citation.resolver._verify_relevance",
                return_value=({"checked": False, "reason": "no key"}, None),
            ),
        ):
            return resolver.resolve_citations(
                [{"claim": "c", "known_urls": [self._REDIRECTOR]}], []
            )[0]

    def test_the_resolved_url_is_recorded_when_it_differs(self):
        assert self._resolve(self._REAL)["final_url"] == self._REAL

    def test_the_requested_url_is_still_recorded(self):
        """The citation as given still has to be reportable — it is what the
        model actually produced."""
        assert self._resolve(self._REAL)["url"] == self._REDIRECTOR

    def test_an_ordinary_citation_gains_no_extra_field(self):
        assert "final_url" not in self._resolve(self._REDIRECTOR)

    def test_the_wayback_fallback_path_does_not_raise(self):
        """`final_url` is assigned inside the try that the fallback skips.
        Leaving it unbound turned every fallback into an unresolved citation --
        caught here rather than by the fallback tests noticing collateral damage.
        """
        with (
            patch(
                "ci_article_review.adapters.citation.resolver.safe_get",
                side_effect=requests.exceptions.ConnectionError("dns"),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                return_value={"archived": False},
            ),
        ):
            result = resolver.resolve_citations(
                [{"claim": "c", "known_urls": [self._REDIRECTOR]}], []
            )[0]
        assert "final_url" not in result
        assert isinstance(result, dict)


class TestResolvedUrlIsWhatTheReaderSees:
    def test_the_report_links_the_resolved_url_not_the_redirector(self):
        from ci_article_review.report_markdown import _render_archive_pair

        out = "\n".join(
            _render_archive_pair(
                {
                    "url": "https://redirector.example/grounding-api-redirect/AUZIYabc",
                    "final_url": "https://www.jalopnik.com/honda-clocks-stuck",
                    "wayback": {"archived": False},
                }
            )
        )
        assert "jalopnik.com" in out
        assert "grounding-api-redirect" not in out

    def test_it_falls_back_to_the_requested_url(self):
        from ci_article_review.report_markdown import _render_archive_pair

        out = "\n".join(
            _render_archive_pair(
                {"url": "https://example.org/a", "wayback": {"archived": False}}
            )
        )
        assert "https://example.org/a" in out
