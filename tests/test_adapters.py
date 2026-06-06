"""Unit tests for adapter modules."""
import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# LanguageTool adapter
# ---------------------------------------------------------------------------

class TestLanguageTool:
    def _make_match(self, offset, length, category_id, replacement, rule_id="RULE"):
        return {
            "offset": offset,
            "length": length,
            "message": "Test message",
            "rule": {"id": rule_id, "category": {"id": category_id}},
            "replacements": [{"value": replacement}],
            "context": {"text": "context text"},
        }

    def test_apply_corrections_auto_apply(self):
        from adapters.grammar.languagetool import apply_corrections
        text = "Teh quick brown fox"
        matches = [self._make_match(0, 3, "TYPOS", "The")]
        corrected, log = apply_corrections(text, matches, {"TYPOS"}, set())
        assert corrected == "The quick brown fox"
        assert len(log) == 1
        assert log[0]["original"] == "Teh"
        assert log[0]["replacement"] == "The"

    def test_apply_corrections_suppressed(self):
        from adapters.grammar.languagetool import apply_corrections
        text = "Running fast."
        matches = [self._make_match(0, 7, "SENTENCE_FRAGMENT", "Run")]
        corrected, log = apply_corrections(text, matches, {"TYPOS"}, {"SENTENCE_FRAGMENT"})
        assert corrected == text
        assert len(log) == 0

    def test_apply_corrections_not_auto_apply_category(self):
        from adapters.grammar.languagetool import apply_corrections
        text = "Some text here."
        matches = [self._make_match(0, 4, "STYLE", "Different")]
        corrected, log = apply_corrections(text, matches, {"TYPOS"}, set())
        assert corrected == text
        assert len(log) == 0

    def test_apply_corrections_multiple_reverse_order(self):
        from adapters.grammar.languagetool import apply_corrections
        text = "Teh cat sat on teh mat"
        matches = [
            self._make_match(0, 3, "TYPOS", "The"),
            self._make_match(15, 3, "TYPOS", "the"),
        ]
        corrected, log = apply_corrections(text, matches, {"TYPOS"}, set())
        assert "Teh" not in corrected
        assert "teh" not in corrected

    def test_run_languagetool_failure(self):
        from adapters.grammar.languagetool import run
        lt_config = {"auto_apply": ["TYPOS"], "flag_for_review": [], "suppress": []}
        with patch("adapters.grammar.languagetool.check_text", side_effect=Exception("API down")):
            result = run("Test text", lt_config, "user@example.com", "key", retry=False)
        assert result["failed"] is True
        assert result["corrected_text"] == "Test text"

    def test_run_languagetool_success(self):
        from adapters.grammar.languagetool import run
        lt_config = {"auto_apply": ["TYPOS"], "flag_for_review": ["STYLE"], "suppress": []}
        mock_response = {
            "matches": [
                {
                    "offset": 0, "length": 3,
                    "message": "Spelling",
                    "rule": {"id": "SPELL", "category": {"id": "TYPOS"}},
                    "replacements": [{"value": "The"}],
                    "context": {"text": "Teh quick"},
                }
            ]
        }
        with patch("adapters.grammar.languagetool.check_text", return_value=mock_response):
            result = run("Teh quick", lt_config, "user@example.com", "key")
        assert result["failed"] is False
        assert result["corrected_text"] == "The quick"
        assert len(result["change_log"]) == 1


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------

class TestOpenAI:
    def _mock_response(self, content_dict):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {
            "choices": [{"message": {"content": json.dumps(content_dict)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock.raise_for_status = MagicMock()
        return mock

    def test_successful_call(self):
        from adapters.review import openai as oai
        content = {"flags": [{"passage": "test", "problem": "hedging", "suggested_rewrite": "direct"}], "low_confidence": []}
        with patch("adapters.review.openai.requests.post", return_value=self._mock_response(content)):
            result = oai.call("system", "user", "key")
        assert result["failed"] is False
        assert result["data"]["flags"][0]["passage"] == "test"
        assert result["tokens"]["prompt"] == 100

    def test_failed_call(self):
        from adapters.review import openai as oai
        with patch("adapters.review.openai.requests.post", side_effect=Exception("Connection error")):
            result = oai.call("system", "user", "key", retry=False)
        assert result["failed"] is True

    def test_malformed_json(self):
        from adapters.review import openai as oai
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {
            "choices": [{"message": {"content": "not json at all"}}],
            "usage": {},
        }
        mock.raise_for_status = MagicMock()
        with patch("adapters.review.openai.requests.post", return_value=mock):
            result = oai.call("system", "user", "key")
        assert result["failed"] is True
        assert result["raw"] == "not json at all"


# ---------------------------------------------------------------------------
# Mistral adapter
# ---------------------------------------------------------------------------

class TestMistral:
    def _mock_response(self, content_dict):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {
            "choices": [{"message": {"content": json.dumps(content_dict)}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 40},
        }
        mock.raise_for_status = MagicMock()
        return mock

    def test_successful_call(self):
        from adapters.review import mistral
        content = {"flags": [], "low_confidence": []}
        with patch("adapters.review.mistral.requests.post", return_value=self._mock_response(content)):
            result = mistral.call("system", "user", "key")
        assert result["failed"] is False
        assert "flags" in result["data"]

    def test_failed_call(self):
        from adapters.review import mistral
        with patch("adapters.review.mistral.requests.post", side_effect=Exception("Timeout")):
            result = mistral.call("system", "user", "key", retry=False)
        assert result["failed"] is True


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------

class TestGemini:
    def _mock_response(self, content_dict):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(content_dict)}]}}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 60},
        }
        mock.raise_for_status = MagicMock()
        return mock

    def test_successful_call(self):
        from adapters.review import gemini
        content = {"confirmed": [], "outdated": [], "contradicted": [], "unverifiable": [], "primary_source_needed": []}
        with patch("adapters.review.gemini.requests.post", return_value=self._mock_response(content)):
            result = gemini.call("system", "user", "key")
        assert result["failed"] is False
        assert "confirmed" in result["data"]

    def test_no_candidates(self):
        from adapters.review import gemini
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"candidates": [], "usageMetadata": {}}
        mock.raise_for_status = MagicMock()
        with patch("adapters.review.gemini.requests.post", return_value=mock):
            result = gemini.call("system", "user", "key")
        assert result["failed"] is True
