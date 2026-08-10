"""Tests for adapters.citation.resolver — parallel resolution, ordering, pointer flag."""

from unittest.mock import MagicMock, patch

import requests

from ci_core import extract

from ci_article_review.adapters.citation import resolver


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
                "ci_article_review.adapters.citation.resolver.requests.get",
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
            "ci_article_review.adapters.citation.resolver.requests.get",
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
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call"
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
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
                return_value={
                    "failed": False,
                    "data": {"verdict": "supports", "reason": "matches"},
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
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
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
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
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
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=mock_resp,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
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

        def fake_call(system, user, api_key, model=None):
            captured["user"] = user
            return {
                "failed": False,
                "data": {"verdict": verdict, "reason": "r"},
                "model": "mistral-small-latest",
                "tokens": {},
                "elapsed_seconds": 0.1,
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=_page_response(html),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call",
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

        def fake_call(system, user, api_key, model=None):
            captured["user"] = user
            return {
                "failed": False,
                "data": {"verdict": "supports", "reason": "r"},
                "model": "mistral-small-latest",
                "tokens": {},
                "elapsed_seconds": 0.1,
            }

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
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
                "ci_article_review.adapters.citation.resolver.mistral.call",
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
                "ci_article_review.adapters.citation.resolver.requests.get",
                return_value=_page_response(self._WALL),
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.wayback.check",
                side_effect=_no_wayback,
            ),
            patch(
                "ci_article_review.adapters.citation.resolver.mistral.call"
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
    """A direct 403 on a known_url should fall back to a Wayback snapshot;
    404/5xx should not — see resolver._wayback_fallback_content scoping."""

    def test_403_recovers_via_wayback_snapshot(self):
        snapshot_url = (
            "https://web.archive.org/web/20240101000000/https://example.com/page"
        )
        # The archived copy is HTML too (with archive.org's banner on top), so
        # it goes through the same article extraction as a direct fetch.
        snap_resp = _page_response()

        with (
            patch(
                "ci_article_review.adapters.citation.resolver.requests.get",
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
                "ci_article_review.adapters.citation.resolver.requests.get",
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
                "ci_article_review.adapters.citation.resolver.requests.get",
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


class TestArchiveSubmission:
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
