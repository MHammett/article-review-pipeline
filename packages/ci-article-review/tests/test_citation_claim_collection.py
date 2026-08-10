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
            fact_check, {"c1": "https://grounded.example"}
        )
        assert claim["known_url"] == "https://own.example"

    def test_grounded_url_is_used_when_the_claim_names_none(self):
        """Finding 4 — a live-search URL is better than nothing."""
        fact_check = {"unverifiable": [{"claim": "c1"}]}
        (claim,) = pipeline._collect_citation_claims(
            fact_check, {"c1": "https://grounded.example"}
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
        assert got == {"c1": "https://cited.example"}

    def test_search_results_are_captured_too(self):
        got = pipeline._collect_grounded_urls(
            self._result(search_results=[{"url": "https://sr.example", "title": "t"}])
        )
        assert got == {"c1": "https://sr.example"}

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
