"""Unit tests for history_analytics — cross-run aggregation over pipeline_history/."""

import json

from ci_article_review import history_analytics as ha


def _write_report(root, slug, run_number, ts, report, filename=None):
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    filename = filename or f"run_{run_number}_{ts.replace(':', '').replace('-', '')}_report.json"
    path = d / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return path


def _api_call(pass_key, failed):
    return {"pass": pass_key, "model": pass_key.split(":")[0], "failed": failed}


def _report(
    generated,
    run_number=1,
    article_title="Test Article",
    api_call_log=None,
    cost_usd=None,
    fk_grade=None,
    seo_issue_count=None,
    broken_links=None,
):
    report = {
        "generated": generated,
        "run_number": run_number,
        "article_title": article_title,
        "publication": "test_pub",
        "model_failures": [c["pass"] for c in (api_call_log or []) if c["failed"]],
        "api_call_log": api_call_log or [],
    }
    if cost_usd is not None:
        report["cost_summary"] = {"total_usd": cost_usd}
    pre_analysis = {}
    if fk_grade is not None:
        pre_analysis["readability"] = {"flesch_kincaid_grade": fk_grade}
    if seo_issue_count is not None:
        pre_analysis["seo"] = {"issues": [{"type": "x"} for _ in range(seo_issue_count)]}
    if broken_links is not None:
        pre_analysis["links"] = [{"ok": i >= broken_links} for i in range(max(broken_links, 1))]
    if pre_analysis:
        report["pre_analysis"] = pre_analysis
    return report


class TestLoadReports:
    def test_loads_and_sorts_chronologically(self, tmp_path):
        _write_report(tmp_path, "article-a", 1, "20260101_000000", _report("2026-01-02T00:00:00+00:00"))
        _write_report(tmp_path, "article-a", 2, "20260101_010000", _report("2026-01-01T00:00:00+00:00"))
        entries = ha.load_reports(tmp_path)
        assert len(entries) == 2
        assert entries[0]["timestamp"] < entries[1]["timestamp"]

    def test_skips_malformed_json(self, tmp_path):
        d = tmp_path / "article-a"
        d.mkdir()
        (d / "run_1_bad_report.json").write_text("{not valid json", encoding="utf-8")
        _write_report(tmp_path, "article-a", 2, "20260101_010000", _report("2026-01-01T00:00:00+00:00"))
        entries = ha.load_reports(tmp_path)
        assert len(entries) == 1

    def test_missing_history_root_returns_empty(self, tmp_path):
        entries = ha.load_reports(tmp_path / "does_not_exist")
        assert entries == []

    def test_scoped_to_one_article(self, tmp_path):
        _write_report(tmp_path, "article-a", 1, "20260101_000000", _report("2026-01-01T00:00:00+00:00"))
        _write_report(tmp_path, "article-b", 1, "20260101_000000", _report("2026-01-01T00:00:00+00:00"))
        entries = ha.load_reports(tmp_path, article_slug="article-a")
        assert len(entries) == 1
        assert entries[0]["slug"] == "article-a"

    def test_falls_back_to_mtime_when_generated_missing(self, tmp_path):
        report = _report("2026-01-01T00:00:00+00:00")
        del report["generated"]
        path = _write_report(tmp_path, "article-a", 1, "20260101_000000", report)
        entries = ha.load_reports(tmp_path)
        assert len(entries) == 1
        assert entries[0]["timestamp"] is not None
        assert path.exists()


class TestProviderReliability:
    def test_detects_degraded_provider(self):
        # perplexity: 5 healthy baseline calls, then 5 straight failures (the
        # "401 across every domain" scenario from the outage this is meant to catch).
        calls = [_api_call("perplexity:fact_check", False) for _ in range(5)]
        calls += [_api_call("perplexity:fact_check", True) for _ in range(5)]
        entries = [
            {"slug": "a", "path": None, "report": {"api_call_log": [c]}, "timestamp": i}
            for i, c in enumerate(calls)
        ]
        result = ha.provider_reliability(entries, recent_window=5, min_baseline=3)
        assert result["perplexity"]["degraded"] is True
        assert result["perplexity"]["recent_success_rate"] == 0.0
        assert result["perplexity"]["baseline_success_rate"] == 1.0

    def test_stable_provider_not_flagged(self):
        calls = [_api_call("openai:fact_check", i % 10 == 0) for i in range(10)]
        entries = [
            {"slug": "a", "path": None, "report": {"api_call_log": [c]}, "timestamp": i}
            for i, c in enumerate(calls)
        ]
        result = ha.provider_reliability(entries, recent_window=5, min_baseline=3)
        assert result["openai"]["degraded"] is False

    def test_insufficient_baseline_not_flagged(self):
        calls = [_api_call("grok:fact_check", True) for _ in range(2)]
        entries = [
            {"slug": "a", "path": None, "report": {"api_call_log": [c]}, "timestamp": i}
            for i, c in enumerate(calls)
        ]
        result = ha.provider_reliability(entries, recent_window=5, min_baseline=3)
        assert result["grok"]["degraded"] is False
        assert result["grok"]["baseline_success_rate"] is None

    def test_no_api_call_log_yields_empty(self):
        entries = [{"slug": "a", "path": None, "report": {}, "timestamp": 0}]
        assert ha.provider_reliability(entries) == {}

    def test_pass_without_colon_is_skipped_not_misattributed(self):
        # Some very old reports recorded "pass" as just the domain name with
        # no colon (e.g. "fact_check" instead of "openai:fact_check") — that
        # can't be attributed to a provider and must not show up as one.
        entries = [
            {
                "slug": "a",
                "path": None,
                "report": {"api_call_log": [_api_call("openai:fact_check", False), {"pass": "fact_check", "failed": False}]},
                "timestamp": 0,
            }
        ]
        result = ha.provider_reliability(entries)
        assert set(result.keys()) == {"openai"}

    def test_multiple_providers_tracked_independently(self):
        entries = [
            {
                "slug": "a",
                "path": None,
                "report": {
                    "api_call_log": [
                        _api_call("openai:fact_check", False),
                        _api_call("perplexity:fact_check", True),
                    ]
                },
                "timestamp": 0,
            }
        ]
        result = ha.provider_reliability(entries)
        assert set(result.keys()) == {"openai", "perplexity"}


class TestCostTrend:
    def test_totals_and_average(self):
        entries = [
            {"slug": "a", "path": None, "report": _report("2026-01-01T00:00:00+00:00", cost_usd=1.0), "timestamp": 0},
            {"slug": "a", "path": None, "report": _report("2026-01-02T00:00:00+00:00", cost_usd=3.0), "timestamp": 1},
        ]
        result = ha.cost_trend(entries, recent_window=5)
        assert result["runs"] == 2
        assert result["total_usd"] == 4.0
        assert result["average_usd"] == 2.0

    def test_increasing_trend_detected(self):
        entries = []
        for i in range(3):
            entries.append(
                {"slug": "a", "path": None, "report": _report("2026-01-01T00:00:00+00:00", cost_usd=1.0), "timestamp": i}
            )
        for i in range(3, 6):
            entries.append(
                {"slug": "a", "path": None, "report": _report("2026-01-01T00:00:00+00:00", cost_usd=5.0), "timestamp": i}
            )
        result = ha.cost_trend(entries, recent_window=3)
        assert result["trend"] == "increasing"

    def test_no_cost_data(self):
        entries = [{"slug": "a", "path": None, "report": _report("2026-01-01T00:00:00+00:00"), "timestamp": 0}]
        result = ha.cost_trend(entries)
        assert result["trend"] == "no_data"
        assert result["runs"] == 0

    def test_insufficient_baseline(self):
        entries = [
            {"slug": "a", "path": None, "report": _report("2026-01-01T00:00:00+00:00", cost_usd=1.0), "timestamp": 0}
        ]
        result = ha.cost_trend(entries, recent_window=5)
        assert result["trend"] == "insufficient_history"
        assert result["baseline_average_usd"] is None


class TestQualityTrend:
    def test_per_article_detects_improvement(self):
        entries = [
            {
                "slug": "a",
                "path": None,
                "report": _report("2026-01-01T00:00:00+00:00", run_number=1, fk_grade=14.0, seo_issue_count=3, broken_links=2),
                "timestamp": 0,
            },
            {
                "slug": "a",
                "path": None,
                "report": _report("2026-01-02T00:00:00+00:00", run_number=2, fk_grade=9.0, seo_issue_count=0, broken_links=0),
                "timestamp": 1,
            },
        ]
        result = ha.per_article_quality_trend(entries)
        assert result["a"]["runs"] == 2
        assert result["a"]["fk_grade_trend"] == "improved"
        assert result["a"]["seo_issues_trend"] == "improved"
        assert result["a"]["broken_links_trend"] == "improved"

    def test_per_article_detects_worsening(self):
        entries = [
            {
                "slug": "a",
                "path": None,
                "report": _report("2026-01-01T00:00:00+00:00", fk_grade=8.0),
                "timestamp": 0,
            },
            {
                "slug": "a",
                "path": None,
                "report": _report("2026-01-02T00:00:00+00:00", fk_grade=15.0),
                "timestamp": 1,
            },
        ]
        result = ha.per_article_quality_trend(entries)
        assert result["a"]["fk_grade_trend"] == "worsened"

    def test_single_run_article_has_unchanged_or_unknown_trend(self):
        entries = [
            {
                "slug": "a",
                "path": None,
                "report": _report("2026-01-01T00:00:00+00:00", fk_grade=10.0),
                "timestamp": 0,
            }
        ]
        result = ha.per_article_quality_trend(entries)
        assert result["a"]["runs"] == 1
        assert result["a"]["fk_grade_trend"] == "unchanged"

    def test_missing_pre_analysis_is_unknown_not_error(self):
        entries = [
            {"slug": "a", "path": None, "report": _report("2026-01-01T00:00:00+00:00"), "timestamp": 0},
            {"slug": "a", "path": None, "report": _report("2026-01-02T00:00:00+00:00"), "timestamp": 1},
        ]
        result = ha.per_article_quality_trend(entries)
        assert result["a"]["fk_grade_trend"] == "unknown"

    def test_global_trend_across_articles(self):
        entries = []
        for i in range(3):
            entries.append(
                {
                    "slug": f"article-{i}",
                    "path": None,
                    "report": _report("2026-01-01T00:00:00+00:00", fk_grade=15.0),
                    "timestamp": i,
                }
            )
        for i in range(3, 6):
            entries.append(
                {
                    "slug": f"article-{i}",
                    "path": None,
                    "report": _report("2026-01-01T00:00:00+00:00", fk_grade=8.0),
                    "timestamp": i,
                }
            )
        result = ha.global_quality_trend(entries, recent_window=3)
        assert result["fk_grade"]["trend"] == "improved"


class TestBuildHistoryReport:
    def test_end_to_end_on_disk_fixtures(self, tmp_path):
        _write_report(
            tmp_path,
            "article-a",
            1,
            "20260101_000000",
            _report(
                "2026-01-01T00:00:00+00:00",
                api_call_log=[_api_call("openai:fact_check", False)],
                cost_usd=0.5,
                fk_grade=10.0,
                seo_issue_count=1,
                broken_links=0,
            ),
        )
        _write_report(
            tmp_path,
            "article-a",
            2,
            "20260102_000000",
            _report(
                "2026-01-02T00:00:00+00:00",
                run_number=2,
                api_call_log=[_api_call("openai:fact_check", False)],
                cost_usd=0.7,
                fk_grade=9.0,
                seo_issue_count=0,
                broken_links=0,
            ),
        )
        result = ha.build_history_report(tmp_path)
        assert result["total_reports"] == 2
        assert "openai" in result["provider_reliability"]
        assert result["cost_trend"]["runs"] == 2
        assert result["per_article_quality_trend"]["article-a"]["runs"] == 2

    def test_empty_history_root(self, tmp_path):
        result = ha.build_history_report(tmp_path / "missing")
        assert result["total_reports"] == 0
        assert result["provider_reliability"] == {}
        assert result["cost_trend"]["trend"] == "no_data"

    def test_print_history_report_does_not_error_on_empty(self, tmp_path, capsys):
        result = ha.build_history_report(tmp_path / "missing")
        ha.print_history_report(result)
        captured = capsys.readouterr()
        assert "No report files found" in captured.out

    def test_print_history_report_flags_degraded_provider(self, tmp_path, capsys):
        entries_dir = tmp_path
        for i in range(5):
            _write_report(
                entries_dir,
                "article-a",
                1,
                f"2026010{i+1}_000000",
                _report(
                    f"2026-01-0{i+1}T00:00:00+00:00",
                    api_call_log=[_api_call("perplexity:fact_check", False)],
                ),
            )
        for i in range(5):
            _write_report(
                entries_dir,
                "article-a",
                1,
                f"2026011{i}_000000",
                _report(
                    f"2026-01-1{i}T00:00:00+00:00",
                    api_call_log=[_api_call("perplexity:fact_check", True)],
                ),
            )
        result = ha.build_history_report(tmp_path, recent_window=5)
        ha.print_history_report(result)
        captured = capsys.readouterr()
        assert "DEGRADED" in captured.out
        assert "perplexity" in captured.out
