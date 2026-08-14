"""Which claims reach Pass 3, and which URL each one is resolved against.

Covers audit findings 3, 4, 5 and 5b(ii) — four separate ways the pipeline was
leaving citation verification on the floor:

* Pass 3 was gated on ``citation_sources`` even though the ``known_url`` path
  never reads it, so a publication with no matching adapter lost *all* citation
  verification (finding 3).
* Provider grounded-search URLs were parsed and discarded (finding 4).
* ``confirmed`` claims — the ones that ship as written — were never resolved or
  archived (finding 5).
* ``primary_source_needed`` items name their URL ``best_candidate_source``, and
  the collector read ``source``, so the one bucket whose whole purpose is to
  recommend a source never had its recommendation fetched (finding 5b).
"""

from unittest.mock import patch

import ci_article_review.pipeline as pipeline
from ci_article_review.adapters.citation import resolver


class TestCollectCitationClaims:
    def test_every_fact_check_bucket_contributes_claims(self):
        fact_check = {
            "confirmed": [{"claim": "c1"}],
            "outdated": [{"claim": "c2"}],
            "contradicted": [{"claim": "c3"}],
            "unverifiable": [{"claim": "c4"}],
            "primary_source_needed": [{"claim": "c5"}],
        }
        claims = pipeline._collect_citation_claims(fact_check, {})
        assert {c["claim"] for c in claims} == {"c1", "c2", "c3", "c4", "c5"}

    def test_confirmed_claims_are_resolved_and_carry_their_source(self):
        """Finding 5 — these ship as written, so they matter most."""
        fact_check = {
            "confirmed": [
                {"claim": "c1", "source": "EIA, https://example.org/eia-table"}
            ]
        }
        (claim,) = pipeline._collect_citation_claims(fact_check, {})
        assert claim["known_url"] == "https://example.org/eia-table"
        assert claim["fact_check_bucket"] == "confirmed"

    def test_primary_source_needed_uses_best_candidate_source(self):
        """Finding 5b — the collector previously read the wrong key."""
        fact_check = {
            "primary_source_needed": [
                {"claim": "c1", "best_candidate_source": "https://example.org/report"}
            ]
        }
        (claim,) = pipeline._collect_citation_claims(fact_check, {})
        assert claim["known_url"] == "https://example.org/report"

    def test_a_claims_own_source_beats_a_grounded_url(self):
        """Claim-specific provenance always wins over a response-level one."""
        fact_check = {"outdated": [{"claim": "c1", "source": "https://own.example"}]}
        (claim,) = pipeline._collect_citation_claims(
            fact_check, {pipeline._claim_key("c1"): "https://grounded.example"}
        )
        assert claim["known_url"] == "https://own.example"

    def test_grounded_url_is_used_when_the_claim_names_none(self):
        """Finding 4 — a live-search URL is better than nothing."""
        fact_check = {"unverifiable": [{"claim": "c1"}]}
        (claim,) = pipeline._collect_citation_claims(
            fact_check, {pipeline._claim_key("c1"): "https://grounded.example"}
        )
        assert claim["known_url"] == "https://grounded.example"

    def test_a_claim_raised_by_two_models_is_resolved_once(self):
        fact_check = {
            "outdated": [{"claim": "same"}, {"claim": "same"}],
            "contradicted": [{"claim": "same"}],
        }
        assert len(pipeline._collect_citation_claims(fact_check, {})) == 1

    def test_claims_with_no_text_are_dropped(self):
        fact_check = {"outdated": [{"claim": ""}, {"source": "x"}]}
        assert pipeline._collect_citation_claims(fact_check, {}) == []


class TestCollectGroundedUrls:
    def _result(self, **extra):
        base = {
            "failed": False,
            "data": {"outdated": [{"claim": "c1"}]},
        }
        base.update(extra)
        return {("perplexity", "fact_check"): base}

    def test_perplexity_citations_are_captured(self):
        got = pipeline._collect_grounded_urls(
            self._result(citations=["https://cited.example"])
        )
        assert got == {pipeline._claim_key("c1"): "https://cited.example"}

    def test_search_results_are_captured_too(self):
        got = pipeline._collect_grounded_urls(
            self._result(search_results=[{"url": "https://sr.example", "title": "t"}])
        )
        assert got == {pipeline._claim_key("c1"): "https://sr.example"}

    def test_failed_calls_contribute_nothing(self):
        got = pipeline._collect_grounded_urls(
            self._result(failed=True, citations=["https://cited.example"])
        )
        assert got == {}

    def test_non_fact_check_domains_are_ignored(self):
        """Grounded URLs are evidence for claims, and only fact_check makes them."""
        results = {
            ("perplexity", "voice_style"): {
                "failed": False,
                "citations": ["https://cited.example"],
                "data": {"flags": [{"passage": "p"}]},
            }
        }
        assert pipeline._collect_grounded_urls(results) == {}

    def test_malformed_entries_do_not_raise(self):
        got = pipeline._collect_grounded_urls(
            self._result(citations=[None, 42], search_results=[{"no_url": 1}, "junk"])
        )
        assert got == {}


class TestPassThreeIsNotGatedOnAdapters:
    """Finding 3 — the regression that silently disabled verification."""

    def test_claims_are_resolved_with_an_empty_citation_sources_list(self):
        called = {}

        def _fake_resolve(claims, citation_sources, *a, **kw):
            called["claims"] = claims
            called["sources"] = citation_sources
            return [{"claim": c["claim"], "resolved": True} for c in claims]

        with patch(
            "ci_article_review.adapters.citation.resolver.resolve_citations",
            side_effect=_fake_resolve,
        ):
            report = _run_minimal_pipeline(citation_sources=[])

        assert called.get("sources") == [], "resolve_citations was never called"
        assert report["section_9_citations"], (
            "Section 9 is empty with no adapters configured — the known_url path "
            "does not need them, so this is the finding-3 regression."
        )


class TestFactCheckBucketSurvivesResolution:
    def test_bucket_is_attached_to_the_result(self):
        with patch.object(
            resolver,
            "_resolve_one",
            return_value={"claim": "c1", "resolved": False},
        ):
            (result,) = resolver.resolve_citations(
                [{"claim": "c1", "known_url": None, "fact_check_bucket": "confirmed"}],
                [],
            )
        assert result["fact_check_bucket"] == "confirmed"

    def test_plain_string_claims_still_work(self):
        """The bare-string form is part of resolve_citations' contract."""
        with patch.object(
            resolver, "_resolve_one", return_value={"claim": "c1", "resolved": False}
        ):
            (result,) = resolver.resolve_citations(["c1"], [])
        assert "fact_check_bucket" not in result


class TestUnresolvableAdapterIsAnnounced:
    """Finding 11 — a configured source that can never match said nothing."""

    def setup_method(self):
        resolver._WARNED_ADAPTERS.clear()

    def test_unknown_adapter_warns(self, caplog):
        resolver._resolve_one(
            "a claim", [{"name": "AWS_DOCS", "adapter": "generic_url"}]
        )
        assert "generic_url" in caplog.text
        assert "does not exist" in caplog.text

    def test_missing_adapter_key_warns(self, caplog):
        resolver._resolve_one("a claim", [{"name": "Nameless"}])
        assert "has no 'adapter' key" in caplog.text

    def test_each_adapter_warns_only_once(self, caplog):
        for _ in range(5):
            resolver._resolve_one("a claim", [{"adapter": "generic_url"}])
        assert caplog.text.count("generic_url") == 1, (
            "A config mistake should be reported once, not once per claim."
        )


# ---------------------------------------------------------------------------


def _run_minimal_pipeline(citation_sources):
    """Drive a full draft run with stubs, returning the report."""
    from contextlib import ExitStack

    config = {
        "api_keys": {"mistral": {"api_key": "k"}},
        "pipeline": {"grammar_pass": False, "link_validation": False},
        "publication": {
            "citation_sources": citation_sources,
            "seo_rules": {"content_review": False},
        },
        "delta": {},
        "ensemble": {},
        "models": {"mistral": {"model": "mistral-large-latest"}},
    }
    currency = {
        "warnings": [],
        "notices": [],
        "registry_warning": False,
        "registry_stale": False,
        "registry_date": "",
        "registry_age_days": 0,
    }

    def _fake_domain(model_name, domain, *a, **kw):
        data = (
            {"outdated": [{"claim": "a claim", "source": "https://example.org/x"}]}
            if domain == "fact_check"
            else {"flags": []}
        )
        return {
            "failed": False,
            "data": data,
            "model": "m",
            "tokens": {},
            "elapsed_seconds": 0.1,
            "_model": model_name,
            "_domain": domain,
        }

    with ExitStack() as stack:
        for target, kw in (
            ("load_user_config", {"return_value": {"pipeline": {}}}),
            ("load_publication_config", {"return_value": {}}),
            ("merge_configs", {"return_value": config}),
            ("check_model_currency", {"return_value": currency}),
            ("_run_domain", {"side_effect": _fake_domain}),
            ("_build_assignments", {"return_value": [("mistral", "fact_check")]}),
            ("_build_custom_assignments", {"return_value": ([], {})}),
            # save_run's real return shape — the pipeline indexes into it.
            (
                "hist.save_run",
                {
                    "return_value": {
                        "report_path": None,
                        "corrections_path": None,
                        "markdown_path": None,
                    }
                },
            ),
        ):
            stack.enter_context(patch(f"ci_article_review.pipeline.{target}", **kw))
        stack.enter_context(
            patch(
                "ci_article_review.analysis.seo_suggest.generate",
                return_value=({"status": "skipped", "reason": "t"}, None),
            )
        )
        stack.enter_context(
            patch(
                "ci_article_review.analysis.seo_content.review",
                return_value=({"status": "skipped", "reason": "t"}, None),
            )
        )
        return pipeline.run_draft_pipeline(
            None,
            "pub",
            handoff={"title": "T", "draft": "Body text here.", "run_number": 1},
        )


class TestMarkdownSourceUrls:
    """Models write markdown links; the bare-URL regex swallowed the syntax.

    A real run produced `[www.cbc.ca](https://www.cbc.ca)` in a source field,
    and the fetch went out against a hostname of literally "[www.cbc.ca]".
    """

    def test_a_markdown_link_yields_its_target(self):
        got = pipeline._extract_source_url("See [www.cbc.ca](https://www.cbc.ca/story)")
        assert got == "https://www.cbc.ca/story"

    def test_the_real_malformed_case_from_the_run(self):
        got = pipeline._extract_source_url("[www.cbc.ca](https://www.cbc.ca)")
        assert got == "https://www.cbc.ca"
        assert "[" not in got and "]" not in got

    def test_a_bare_url_still_works(self):
        got = pipeline._extract_source_url("EIA Profile, https://www.eia.gov/state/")
        assert got == "https://www.eia.gov/state/"

    def test_trailing_sentence_punctuation_is_stripped(self):
        assert pipeline._extract_source_url("see https://x.example/a.") == (
            "https://x.example/a"
        )

    def test_no_url_returns_none(self):
        assert pipeline._extract_source_url("EIA State Energy Profile, 2024") is None


class TestGroundedUrlLossIsReported:
    """A failed fact-check pass silently degrades Section 9 — say so.

    Measured 2026-08-12: perplexity:fact_check was rate-limited, grounded-search
    URLs went 18 -> 0, and citation resolution fell from 48% (49/101) to 22%
    (32/144). The run summary reported "perplexity:fact_check FAILED" in one
    place and "9 verified" in another, with nothing connecting them, so the drop
    read as an unexplained regression rather than a known consequence.
    """

    def _results(self, perplexity_failed):
        return {
            ("gemini", "fact_check"): {"failed": False, "data": {}},
            ("perplexity", "fact_check"): (
                {"failed": True, "error": "429 Too Many Requests"}
                if perplexity_failed
                else {
                    "failed": False,
                    "citations": ["https://example.org/source"],
                    "data": {"confirmed": [{"claim": "a claim"}]},
                }
            ),
        }

    def test_grounded_urls_come_only_from_a_successful_fact_check_pass(self):
        assert pipeline._collect_grounded_urls(self._results(False))
        assert pipeline._collect_grounded_urls(self._results(True)) == {}

    def test_failure_that_costs_grounded_urls_is_recorded_as_a_degradation(self):
        report = {"section_2_fact_check": {}}
        degradations = []

        # Reproduce the branch the pipeline takes at Pass 3 without running the
        # whole pass: no grounded URLs, and a fact-check pass that failed.
        results = self._results(True)
        grounded = pipeline._collect_grounded_urls(results)
        failed = sorted(
            f"{m}:{d}"
            for (m, d), r in results.items()
            if d == "fact_check" and r.get("failed") and not r.get("skipped")
        )
        assert grounded == {}
        assert failed == ["perplexity:fact_check"]

        if not grounded and failed:
            degradations.append(
                {"section": "SECTION 9: Citations", "caused_by": failed}
            )
        report["degradations"] = degradations

        assert report["degradations"][0]["caused_by"] == ["perplexity:fact_check"]

    def test_summary_prints_the_degradation_next_to_the_failure(self, capsys):
        from ci_article_review.pipeline import _print_draft_summary

        report = {
            "article_title": "T",
            "run_number": 17,
            "generated": "2026-08-12T00:00:00+00:00",
            "section_1_consensus": [],
            "section_2_fact_check": {},
            "section_3_voice": [],
            "section_4_argument": [],
            "section_5_completeness": [],
            "section_6_red_team": {},
            "section_7_low_confidence": [],
            "lt_corrections_applied": [],
            "lt_skipped": True,
            "model_failures": ["perplexity:fact_check"],
            "degradations": [
                {
                    "section": "SECTION 9: Citations",
                    "caused_by": ["perplexity:fact_check"],
                    "detail": "Section 9 degraded: no grounded-search URLs were "
                    "available because perplexity:fact_check failed.",
                }
            ],
        }
        _print_draft_summary(report, {})
        out = capsys.readouterr().out
        assert "perplexity:fact_check" in out
        assert "Section 9 degraded" in out
        # The link is only useful if both appear together.
        assert out.index("These model passes failed") < out.index("Section 9 degraded")


class TestGeminiGroundingReachesClaims:
    """Both grounded providers must supply URLs, so neither is a single point of failure."""

    def _gemini_result(self):
        return {
            "failed": False,
            "grounding_chunks": [
                {
                    "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAA",
                    "title": "epa.gov",
                }
            ],
            "data": {"outdated": [{"claim": "gemini claim"}]},
        }

    def test_gemini_grounding_resolves_into_a_claim_url(self):
        with patch(
            "ci_article_review.pipeline.gemini_adapter.resolve_grounding_urls",
            return_value=["https://www.epa.gov/superfund/pfoa"],
        ):
            got = pipeline._collect_grounded_urls(
                {("gemini", "fact_check"): self._gemini_result()}
            )
        assert got == {
            pipeline._claim_key("gemini claim"): "https://www.epa.gov/superfund/pfoa"
        }

    def test_perplexity_failing_no_longer_empties_the_grounded_urls(self):
        """The 2026-08-12 cascade: gemini succeeded, perplexity didn't, and we got zero."""
        results = {
            ("perplexity", "fact_check"): {"failed": True, "error": "429"},
            ("gemini", "fact_check"): self._gemini_result(),
        }
        with patch(
            "ci_article_review.pipeline.gemini_adapter.resolve_grounding_urls",
            return_value=["https://www.epa.gov/superfund/pfoa"],
        ):
            got = pipeline._collect_grounded_urls(results)
        assert got, "gemini succeeded, so grounded URLs must not be empty"

    def test_redirects_are_resolved_once_across_all_gemini_domains(self):
        results = {
            ("gemini", "fact_check"): self._gemini_result(),
        }
        with patch(
            "ci_article_review.pipeline.gemini_adapter.resolve_grounding_urls",
            return_value=["https://www.epa.gov/superfund/pfoa"],
        ) as mock_resolve:
            pipeline._collect_grounded_urls(results)
        # A single shared cache dict must be threaded through every call.
        assert mock_resolve.call_args.kwargs.get("cache") is not None

    def test_a_wrapper_url_never_reaches_a_claim(self):
        """Storing an expiring redirect is worse than storing nothing."""
        with patch(
            "ci_article_review.pipeline.gemini_adapter.resolve_grounding_urls",
            return_value=[],
        ):
            got = pipeline._collect_grounded_urls(
                {("gemini", "fact_check"): self._gemini_result()}
            )
        assert got == {}


class TestClaimDeduplicationByMeaning:
    """Five models paraphrasing one fact must produce one claim, not five.

    The 2026-08-12 run carried 29 near-duplicate pairs among 144 claims — one
    differing from its twin only by a trailing full stop. Each duplicate bought
    its own resolution fetch, verification call, and Section 9 line.
    """

    def _claims(self, *texts):
        return pipeline._collect_citation_claims(
            {"outdated": [{"claim": t} for t in texts]}, {}
        )

    def test_a_trailing_period_is_not_a_new_claim(self):
        got = self._claims(
            "Microsoft dropped its NDA requirements in March 2026",
            "Microsoft dropped its NDA requirements in March 2026.",
        )
        assert len(got) == 1

    def test_a_leading_article_is_not_a_new_claim(self):
        got = self._claims(
            "The Virginia electrical grid runs below the national average",
            "Virginia electrical grid runs below the national average",
        )
        assert len(got) == 1

    def test_case_and_punctuation_differences_collapse(self):
        got = self._claims(
            "EPA designated PFOA as a CERCLA hazardous substance in 2024",
            "EPA designated PFOA as a CERCLA hazardous substance in 2024!",
        )
        assert len(got) == 1

    def test_genuinely_different_claims_both_survive(self):
        """Merging two distinct claims silently drops one from verification."""
        got = self._claims(
            "US data centers consumed 17.4 billion gallons of water in 2023",
            "US crop irrigation withdrawals average roughly 105 billion gallons per day",
        )
        assert len(got) == 2

    def test_claims_differing_only_in_a_number_are_kept_apart(self):
        """The threshold is set high precisely so this case does not merge."""
        got = self._claims(
            "Emergency engines are capped at 100 hours per year",
            "Emergency engines are capped at 500 hours per year",
        )
        assert len(got) == 2, [c["claim"] for c in got]

    def test_the_first_occurrence_wins_and_keeps_its_url(self):
        claims = pipeline._collect_citation_claims(
            {
                "outdated": [
                    {"claim": "A fact about water", "source": "https://first.example"},
                    {
                        "claim": "A fact about water.",
                        "source": "https://second.example",
                    },
                ]
            },
            {},
        )
        assert len(claims) == 1
        assert claims[0]["known_url"] == "https://first.example"

    def test_an_empty_claim_never_matches_anything(self):
        assert pipeline._is_duplicate_claim(frozenset(), [frozenset()]) is False


class TestGroundedUrlsSurviveDeduplication:
    """Dedup and grounded-URL lookup must agree on what "the same claim" means.

    Regression guard: when claim dedup started collapsing paraphrases but the
    grounded-URL map was still keyed on raw text, a URL registered under the
    phrasing that lost the tie became unreachable by the phrasing that won it —
    silently costing the claim its only source.
    """

    def test_a_url_registered_under_a_variant_phrasing_is_still_found(self):
        grounded = pipeline._collect_grounded_urls(
            {
                ("perplexity", "fact_check"): {
                    "failed": False,
                    "citations": ["https://source.example"],
                    "data": {
                        "outdated": [
                            {
                                "claim": "Microsoft dropped its NDA requirements in March 2026."
                            }
                        ]
                    },
                }
            }
        )
        claims = pipeline._collect_citation_claims(
            {
                "outdated": [
                    {"claim": "Microsoft dropped its NDA requirements in March 2026"},
                    {"claim": "Microsoft dropped its NDA requirements in March 2026."},
                ]
            },
            grounded,
        )
        assert len(claims) == 1
        assert claims[0]["known_url"] == "https://source.example"

    def test_the_grounded_map_is_keyed_the_same_way_dedup_is(self):
        grounded = pipeline._collect_grounded_urls(
            {
                ("perplexity", "fact_check"): {
                    "failed": False,
                    "citations": ["https://source.example"],
                    "data": {"outdated": [{"claim": "The grid runs below average"}]},
                }
            }
        )
        assert list(grounded) == [pipeline._claim_key("The grid runs below average")]
