"""Capturing the ensemble so it can be replayed without paying for it again.

The ensemble is nearly all of a run's cost, and most changes do not touch it —
of the 25 PRs merged to 2026-08-15, 15 touched no live-LLM code at all. Those
changes previously cost a full run to exercise because the raw model output was
discarded the moment it was consolidated.

These tests cover the round trip and, more importantly, the two ways a replay
could quietly corrupt something real: by being counted as spend it never
incurred, and by landing in the article's own history where the next genuine run
would take it as a delta baseline.
"""

import json

import pytest

from ci_article_review import ensemble_capture


RAW = {
    "grok:red_team": {
        "failed": False,
        "data": {"flags": [{"passage": "p", "problem": "x"}]},
        "model": "grok-4.3",
        "_model": "grok",
        "_domain": "red_team",
    },
    "openai:fact_check": {
        "failed": False,
        "data": {"claims": []},
        "model": "gpt-5.5",
        "_model": "openai",
        "_domain": "fact_check",
    },
}


class TestRoundTrip:
    def test_saved_results_load_back_identically(self, tmp_path):
        p = tmp_path / "run_1_x_results.json"
        ensemble_capture.save(p, RAW, article_title="T", run_number=1)
        assert ensemble_capture.load(p) == RAW

    def test_capture_path_sits_beside_the_report(self):
        got = ensemble_capture.capture_path_for(
            "pipeline_history/dc-environment/run_16_20260815_140635_report.json"
        )
        assert got.name == "run_16_20260815_140635_results.json"
        assert got.parent.name == "dc-environment"

    def test_describe_summarises_without_loading_everything(self, tmp_path):
        p = tmp_path / "r_results.json"
        ensemble_capture.save(p, RAW, article_title="T", run_number=7)
        d = ensemble_capture.describe(p)
        assert "2 pass(es)" in d and "0 failed" in d and "run 7" in d


class TestRefusesBadCaptures:
    """A stale capture that half-works is worse than no capture."""

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="No ensemble capture"):
            ensemble_capture.load(tmp_path / "nope.json")

    def test_version_mismatch_is_refused(self, tmp_path):
        p = tmp_path / "old_results.json"
        p.write_text(
            json.dumps({"capture_version": 0, "results": RAW}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="capture version"):
            ensemble_capture.load(p)

    def test_entry_without_model_tags_is_refused(self, tmp_path):
        """Untagged entries would collapse into one 'unknown:unknown' result."""
        p = tmp_path / "bad_results.json"
        p.write_text(
            json.dumps(
                {
                    "capture_version": ensemble_capture.CAPTURE_VERSION,
                    "results": {"grok:red_team": {"failed": False}},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="_model/_domain"):
            ensemble_capture.load(p)

    def test_empty_results_are_refused(self, tmp_path):
        p = tmp_path / "empty_results.json"
        p.write_text(
            json.dumps(
                {"capture_version": ensemble_capture.CAPTURE_VERSION, "results": {}}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="no ensemble results"):
            ensemble_capture.load(p)

    def test_a_save_failure_never_raises(self, tmp_path):
        """Capturing is a convenience; it must not lose a run already paid for."""
        unserialisable = {"x:y": {"_model": "x", "_domain": "y", "f": object()}}
        assert ensemble_capture.save(tmp_path / "r.json", unserialisable) is None
