"""Unit tests for the consolidation module."""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consolidation import _find_consensus, _passage_key, build_report, rerun_recommended


class TestPassageKey:
    def test_normalizes_whitespace(self):
        assert _passage_key("hello   world") == _passage_key("hello world")

    def test_lowercased(self):
        assert _passage_key("Hello World") == _passage_key("hello world")

    def test_truncated_at_120(self):
        long = "a" * 200
        assert len(_passage_key(long)) == 120


class TestFindConsensus:
    def _flag(self, passage, extra=None):
        f = {"passage": passage, "problem": "test"}
        if extra:
            f.update(extra)
        return f

    def test_three_model_consensus(self):
        model_flags = {
            "openai_voice": [self._flag("The government should utilize more resources")],
            "mistral_argument": [self._flag("The government should utilize more resources")],
            "openai_completeness": [self._flag("The government should utilize more resources")],
        }
        consensus, single = _find_consensus(model_flags, [])
        assert len(consensus) == 1
        assert len(single) == 0
        assert set(consensus[0]["models"]) == {"openai_voice", "mistral_argument", "openai_completeness"}

    def test_two_model_plus_lt(self):
        passage = "It is worth noting that costs have risen"
        model_flags = {
            "openai_voice": [self._flag(passage)],
            "mistral_argument": [self._flag(passage)],
        }
        lt_flagged = [passage]
        consensus, single = _find_consensus(model_flags, lt_flagged)
        assert len(consensus) == 1
        assert consensus[0]["languagetool_also_flagged"] is True

    def test_single_model_goes_to_low_confidence(self):
        model_flags = {
            "openai_voice": [self._flag("Some minor phrasing issue here")],
        }
        consensus, single = _find_consensus(model_flags, [])
        assert len(consensus) == 0
        assert len(single) == 1

    def test_empty_passage_excluded(self):
        model_flags = {
            "openai_voice": [{"passage": "", "problem": "empty"}],
            "mistral_argument": [{"passage": "", "problem": "empty"}],
            "openai_completeness": [{"passage": "", "problem": "empty"}],
        }
        consensus, single = _find_consensus(model_flags, [])
        assert len(consensus) == 0


class TestBuildReport:
    def _failed_result(self, pass_name="test"):
        return {"failed": True, "error": "test error", "pass": pass_name, "model": "unknown", "tokens": {}}

    def _ok_result(self, data, pass_name="test", model="test-model"):
        return {"failed": False, "data": data, "pass": pass_name, "model": model, "tokens": {"prompt": 10, "completion": 5}}

    def test_basic_report_structure(self):
        report = build_report(
            article_title="Test Article",
            publication_name="test_pub",
            run_number=1,
            corrected_draft="Test draft content.",
            lt_result={"change_log": [], "flagged_matches": [], "failed": False, "corrected_text": "Test draft content."},
            gemini_result=self._ok_result({"confirmed": [], "outdated": [], "contradicted": [], "unverifiable": [], "primary_source_needed": []}),
            openai_voice_result=self._ok_result({"flags": [], "low_confidence": []}),
            mistral_argument_result=self._ok_result({"flags": [], "low_confidence": []}),
            openai_completeness_result=self._ok_result({"flags": [], "low_confidence": []}),
            mistral_redteam_result=self._ok_result({}),
            api_call_log=[],
        )
        assert report["article_title"] == "Test Article"
        assert report["run_number"] == 1
        assert "section_1_consensus" in report
        assert "section_6_red_team" in report
        assert report["model_failures"] == []

    def test_failed_models_logged(self):
        report = build_report(
            article_title="Test",
            publication_name="pub",
            run_number=1,
            corrected_draft="draft",
            lt_result={"change_log": [], "flagged_matches": [], "failed": False},
            gemini_result=self._failed_result("gemini_fact_check"),
            openai_voice_result=self._failed_result("openai_voice"),
            mistral_argument_result=self._ok_result({"flags": [], "low_confidence": []}),
            openai_completeness_result=self._ok_result({"flags": [], "low_confidence": []}),
            mistral_redteam_result=self._ok_result({}),
            api_call_log=[],
        )
        assert "gemini_fact_check" in report["model_failures"]
        assert "openai_voice" in report["model_failures"]

    def test_lt_failure_flagged(self):
        report = build_report(
            article_title="Test",
            publication_name="pub",
            run_number=1,
            corrected_draft="draft",
            lt_result={"failed": True, "error": "API down", "change_log": []},
            gemini_result=self._failed_result(),
            openai_voice_result=self._failed_result(),
            mistral_argument_result=self._failed_result(),
            openai_completeness_result=self._failed_result(),
            mistral_redteam_result=self._failed_result(),
            api_call_log=[],
        )
        assert report["lt_failed"] is True


class TestRerunRecommended:
    def test_high_word_change(self):
        delta = {"word_change_pct": 20, "new_consensus_count": 0, "resolved_consensus_count": 0,
                 "prior_consensus_count": 5, "current_consensus_count": 5}
        assert rerun_recommended(delta, {"word_change_threshold_pct": 15}) is True

    def test_new_consensus_flags(self):
        delta = {"word_change_pct": 5, "new_consensus_count": 2, "resolved_consensus_count": 1,
                 "prior_consensus_count": 3, "current_consensus_count": 4}
        assert rerun_recommended(delta, {"word_change_threshold_pct": 15}) is True

    def test_stable_draft(self):
        delta = {"word_change_pct": 5, "new_consensus_count": 0, "resolved_consensus_count": 2,
                 "prior_consensus_count": 3, "current_consensus_count": 1}
        assert rerun_recommended(delta, {"word_change_threshold_pct": 15}) is False

    def test_no_delta(self):
        assert rerun_recommended(None, {}) is False
