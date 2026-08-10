"""Unit tests for voice_pattern_report — recurring voice pattern detection."""

import json

from ci_article_review import voice_pattern_report as vpr


def _write_report(root, slug, run_number, ts, report):
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    filename = f"run_{run_number}_{ts}_report.json"
    path = d / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return path


def _voice_flag(passage, problem, source_model="openai", suggested_rewrite=""):
    return {
        "passage": passage,
        "problem": problem,
        "suggested_rewrite": suggested_rewrite,
        "source_model": source_model,
    }


def _report(
    article_title="Test Article",
    publication="test_pub",
    section_3_voice=None,
    section_8_additional=None,
    generated="2026-01-01T00:00:00+00:00",
):
    return {
        "generated": generated,
        "article_title": article_title,
        "publication": publication,
        "section_3_voice": section_3_voice or [],
        "section_8_additional": section_8_additional or [],
    }


REPEATED_PASSAGE = "It's important to note that the industry is not blameless here."
REPEATED_PROBLEM = (
    "This is a familiar editorial hedging phrase that weakens the technical tone."
)


class TestExtractVoiceFindings:
    def test_extracts_section_3_voice(self, tmp_path):
        _write_report(
            tmp_path,
            "article-a",
            1,
            "20260101_000000",
            _report(section_3_voice=[_voice_flag("some passage", "some problem")]),
        )
        from ci_article_review.history_analytics import load_reports

        entries = load_reports(tmp_path)
        findings = vpr.extract_voice_findings(entries)
        assert len(findings) == 1
        assert findings[0]["passage"] == "some passage"
        assert findings[0]["problem"] == "some problem"
        assert findings[0]["section"] == "section_3_voice"

    def test_extracts_section_8_additional_voice_category_only(self, tmp_path):
        from ci_article_review.history_analytics import load_reports

        _write_report(
            tmp_path,
            "article-a",
            1,
            "20260101_000000",
            _report(
                section_8_additional=[
                    {
                        "category": "voice",
                        "passage": "voice-flagged passage",
                        "observation": "voice observation",
                        "source_model": "claude",
                    },
                    {
                        "category": "fact_check",
                        "passage": "unrelated",
                        "observation": "unrelated observation",
                        "source_model": "claude",
                    },
                ]
            ),
        )
        entries = load_reports(tmp_path)
        findings = vpr.extract_voice_findings(entries)
        assert len(findings) == 1
        assert findings[0]["passage"] == "voice-flagged passage"
        assert findings[0]["problem"] == "voice observation"
        assert findings[0]["section"] == "section_8_additional"

    def test_filters_by_publication(self, tmp_path):
        from ci_article_review.history_analytics import load_reports

        _write_report(
            tmp_path,
            "article-a",
            1,
            "20260101_000000",
            _report(
                publication="pub_a",
                section_3_voice=[_voice_flag("p1", "problem1")],
            ),
        )
        _write_report(
            tmp_path,
            "article-b",
            1,
            "20260101_000000",
            _report(
                publication="pub_b",
                section_3_voice=[_voice_flag("p2", "problem2")],
            ),
        )
        entries = load_reports(tmp_path)
        findings = vpr.extract_voice_findings(entries, publication="pub_a")
        assert len(findings) == 1
        assert findings[0]["passage"] == "p1"


class TestClusteringAndCandidates:
    def _findings_for(self, slug, passage, problem, source_model="openai"):
        return {
            "slug": slug,
            "article_title": slug,
            "passage": passage,
            "problem": problem,
            "suggested_rewrite": "",
            "source_model": source_model,
            "section": "section_3_voice",
        }

    def test_recurring_pattern_across_articles_is_flagged(self):
        findings = [
            self._findings_for("article-1", REPEATED_PASSAGE, "problem A"),
            self._findings_for("article-2", REPEATED_PASSAGE, "problem B"),
            self._findings_for("article-3", REPEATED_PASSAGE, "problem C"),
        ]
        candidates = vpr.candidate_patterns(findings, "passage", min_articles=3)
        assert len(candidates) == 1
        assert candidates[0]["distinct_article_count"] == 3
        assert set(candidates[0]["articles"]) == {"article-1", "article-2", "article-3"}

    def test_near_duplicate_wording_still_clusters(self):
        findings = [
            self._findings_for(
                "article-1", "It's worth noting that this matters a lot.", "p"
            ),
            self._findings_for(
                "article-2", "It's worth noting that this really matters.", "p"
            ),
            self._findings_for(
                "article-3", "It's worth noting that this matters quite a bit.", "p"
            ),
        ]
        candidates = vpr.candidate_patterns(
            findings, "passage", min_articles=3, similarity_threshold=0.75
        )
        assert len(candidates) == 1
        assert candidates[0]["distinct_article_count"] == 3

    def test_single_article_finding_not_flagged(self):
        findings = [
            self._findings_for("article-1", REPEATED_PASSAGE, "problem A"),
            self._findings_for("article-1", REPEATED_PASSAGE, "problem A"),
        ]
        candidates = vpr.candidate_patterns(findings, "passage", min_articles=3)
        assert candidates == []

    def test_below_threshold_two_articles_not_flagged_with_default_min(self):
        findings = [
            self._findings_for("article-1", REPEATED_PASSAGE, "problem A"),
            self._findings_for("article-2", REPEATED_PASSAGE, "problem B"),
        ]
        candidates = vpr.candidate_patterns(findings, "passage", min_articles=3)
        assert candidates == []

    def test_unrelated_findings_do_not_cluster_together(self):
        findings = [
            self._findings_for("article-1", "Completely unrelated text one.", "p"),
            self._findings_for("article-2", "Something else entirely different.", "p"),
            self._findings_for("article-3", "Yet another distinct sentence here.", "p"),
        ]
        candidates = vpr.candidate_patterns(findings, "passage", min_articles=3)
        assert candidates == []

    def test_min_articles_counts_distinct_articles_not_occurrences(self):
        # Same pattern hit 5 times but only across 2 distinct articles —
        # must not qualify under a min_articles=3 threshold.
        findings = [
            self._findings_for("article-1", REPEATED_PASSAGE, "p"),
            self._findings_for("article-1", REPEATED_PASSAGE, "p"),
            self._findings_for("article-1", REPEATED_PASSAGE, "p"),
            self._findings_for("article-2", REPEATED_PASSAGE, "p"),
            self._findings_for("article-2", REPEATED_PASSAGE, "p"),
        ]
        candidates = vpr.candidate_patterns(findings, "passage", min_articles=3)
        assert candidates == []

    def test_already_banned_pattern_is_excluded(self):
        findings = [
            self._findings_for("article-1", "at the end of the day", "p"),
            self._findings_for("article-2", "at the end of the day", "p"),
            self._findings_for("article-3", "at the end of the day", "p"),
        ]
        candidates = vpr.candidate_patterns(
            findings,
            "passage",
            min_articles=3,
            banned_phrases={"at the end of the day"},
        )
        assert candidates == []

    def test_problem_text_clustering_independent_of_passage(self):
        findings = [
            self._findings_for("article-1", "passage one", REPEATED_PROBLEM),
            self._findings_for("article-2", "passage two", REPEATED_PROBLEM),
            self._findings_for("article-3", "passage three", REPEATED_PROBLEM),
        ]
        passage_candidates = vpr.candidate_patterns(findings, "passage", min_articles=3)
        problem_candidates = vpr.candidate_patterns(findings, "problem", min_articles=3)
        assert passage_candidates == []
        assert len(problem_candidates) == 1


class TestLoadBannedTerms:
    def test_missing_config_returns_empty_sets(self):
        words, phrases = vpr.load_banned_terms(None)
        assert words == set()
        assert phrases == set()

    def test_reads_style_rules(self, tmp_path):
        config_path = tmp_path / "pub.yaml"
        config_path.write_text(
            "style_rules:\n"
            "  banned_words:\n"
            "    - delve\n"
            "  banned_phrases:\n"
            "    - shed light on\n",
            encoding="utf-8",
        )
        words, phrases = vpr.load_banned_terms(config_path)
        assert words == {"delve"}
        assert phrases == {"shed light on"}


class TestBuildVoicePatternReport:
    def test_end_to_end_on_disk_fixtures(self, tmp_path):
        for i in range(3):
            _write_report(
                tmp_path,
                f"article-{i}",
                1,
                f"2026010{i + 1}_000000",
                _report(
                    article_title=f"Article {i}",
                    section_3_voice=[_voice_flag(REPEATED_PASSAGE, "problem")],
                ),
            )
        result = vpr.build_voice_pattern_report(tmp_path, min_articles=3)
        assert result["total_reports"] == 3
        assert result["total_voice_findings"] == 3
        assert len(result["passage_candidates"]) == 1
        assert result["passage_candidates"][0]["distinct_article_count"] == 3

    def test_empty_history_root(self, tmp_path):
        result = vpr.build_voice_pattern_report(tmp_path / "missing")
        assert result["total_reports"] == 0
        assert result["total_voice_findings"] == 0
        assert result["passage_candidates"] == []
        assert result["problem_pattern_candidates"] == []

    def test_print_report_does_not_error_on_empty(self, tmp_path, capsys):
        result = vpr.build_voice_pattern_report(tmp_path / "missing")
        vpr.print_voice_pattern_report(result)
        captured = capsys.readouterr()
        assert "No voice findings found" in captured.out

    def test_print_report_shows_candidate(self, tmp_path, capsys):
        for i in range(3):
            _write_report(
                tmp_path,
                f"article-{i}",
                1,
                f"2026010{i + 1}_000000",
                _report(section_3_voice=[_voice_flag(REPEATED_PASSAGE, "problem")]),
            )
        result = vpr.build_voice_pattern_report(tmp_path, min_articles=3)
        vpr.print_voice_pattern_report(result)
        captured = capsys.readouterr()
        assert "Flagged in 3 article(s)" in captured.out
        assert "suggestion report only" in captured.out
