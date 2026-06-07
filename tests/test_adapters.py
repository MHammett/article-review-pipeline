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
        assert "elapsed_seconds" in result

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
        assert "elapsed_seconds" in result


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------

class TestOpenAI:
    def _mock_response(self, content_dict, status=200):
        mock = MagicMock()
        mock.status_code = status
        mock.json.return_value = {
            "choices": [{"message": {"content": json.dumps(content_dict)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock.raise_for_status = MagicMock()
        return mock

    def test_successful_call(self):
        from adapters.review import openai as oai
        content = {"flags": [{"passage": "test", "problem": "hedging", "suggested_rewrite": "direct"}], "low_confidence": []}
        with patch("adapters.review.openai.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = oai.call("system", "user", "key")
        assert result["failed"] is False
        assert result["data"]["flags"][0]["passage"] == "test"
        assert result["tokens"]["prompt"] == 100
        assert "elapsed_seconds" in result

    def test_failed_call(self):
        from adapters.review import openai as oai
        with patch("adapters.review.openai.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("Connection error")
            result = oai.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert "elapsed_seconds" in result

    def test_malformed_json(self):
        from adapters.review import openai as oai
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {
            "choices": [{"message": {"content": "not json at all"}}],
            "usage": {},
        }
        mock.raise_for_status = MagicMock()
        with patch("adapters.review.openai.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = mock
            result = oai.call("system", "user", "key")
        assert result["failed"] is True
        assert result["raw"] == "not json at all"

    def test_model_override(self):
        from adapters.review import openai as oai
        content = {"flags": [], "low_confidence": []}
        with patch("adapters.review.openai.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = oai.call("system", "user", "key", model="gpt-4-turbo")
        assert result["model"] == "gpt-4-turbo"


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
        with patch("adapters.review.mistral.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = mistral.call("system", "user", "key")
        assert result["failed"] is False
        assert "flags" in result["data"]
        assert "elapsed_seconds" in result

    def test_failed_call(self):
        from adapters.review import mistral
        with patch("adapters.review.mistral.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("Timeout")
            result = mistral.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert "elapsed_seconds" in result


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
        with patch("adapters.review.gemini.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = gemini.call("system", "user", "key")
        assert result["failed"] is False
        assert "confirmed" in result["data"]
        assert "elapsed_seconds" in result

    def test_no_candidates(self):
        from adapters.review import gemini
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"candidates": [], "usageMetadata": {}}
        mock.raise_for_status = MagicMock()
        with patch("adapters.review.gemini.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = mock
            result = gemini.call("system", "user", "key")
        assert result["failed"] is True

    def test_api_key_redacted_in_errors(self):
        from adapters.review.gemini import _redact_key
        api_key = "super-secret-key-abc123"
        error_with_key = f"HTTPError at https://example.com?key={api_key} returned 500"
        redacted = _redact_key(error_with_key, api_key)
        assert api_key not in redacted
        assert "[REDACTED]" in redacted

    def test_key_not_redacted_when_absent(self):
        from adapters.review.gemini import _redact_key
        result = _redact_key("Some generic error message", "mykey")
        assert result == "Some generic error message"


# ---------------------------------------------------------------------------
# Grok adapter
# ---------------------------------------------------------------------------

class TestGrok:
    def _mock_response(self, content_dict):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {
            "choices": [{"message": {"content": json.dumps(content_dict)}}],
            "usage": {"prompt_tokens": 90, "completion_tokens": 45},
        }
        mock.raise_for_status = MagicMock()
        return mock

    def test_successful_call(self):
        from adapters.review import grok
        content = {
            "most_vulnerable_claim": {"passage": "test", "attack_vector": "x", "supporting_evidence_for_attack": "y"},
            "highest_audience_risk": {"passage": "test", "risk": "z", "audience_segment": "all"},
            "highest_credibility_risk": {"passage": "test", "risk": "w", "attack_vector": "v"},
        }
        with patch("adapters.review.grok.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = grok.call("system", "user", "key")
        assert result["failed"] is False
        assert "most_vulnerable_claim" in result["data"]
        assert "elapsed_seconds" in result

    def test_failed_call(self):
        from adapters.review import grok
        with patch("adapters.review.grok.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = Exception("Connection refused")
            result = grok.call("system", "user", "key", retry=False)
        assert result["failed"] is True
        assert "elapsed_seconds" in result

    def test_model_override(self):
        from adapters.review import grok
        content = {"most_vulnerable_claim": {}, "highest_audience_risk": {}, "highest_credibility_risk": {}}
        with patch("adapters.review.grok.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.return_value = self._mock_response(content)
            result = grok.call("system", "user", "key", model="grok-2-latest")
        assert result["model"] == "grok-2-latest"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

class TestConfigLoader:
    def test_invalid_publication_name_raises(self):
        from config_loader import validate_publication_name
        with pytest.raises(ValueError, match="Invalid publication name"):
            validate_publication_name("../../etc/shadow")

    def test_invalid_publication_name_with_slash(self):
        from config_loader import validate_publication_name
        with pytest.raises(ValueError):
            validate_publication_name("my/blog")

    def test_valid_publication_name(self):
        from config_loader import validate_publication_name
        validate_publication_name("my-blog")
        validate_publication_name("myblog")
        validate_publication_name("my_blog_2024")

    def test_env_var_missing_gives_helpful_error(self):
        from config_loader import _resolve_env
        import os
        env_key = "PIPELINE_TEST_MISSING_VAR_XYZ"
        if env_key in os.environ:
            del os.environ[env_key]
        with pytest.raises(ValueError, match="not set"):
            _resolve_env(f"${{{env_key}}}")


# ---------------------------------------------------------------------------
# History / slug safety
# ---------------------------------------------------------------------------

class TestHistory:
    def test_slug_strips_special_chars(self):
        from history import _slug
        assert "/" not in _slug("My Article: Part 1/2")
        assert ":" not in _slug("My Article: Part 1/2")

    def test_slug_windows_reserved_names(self):
        from history import _slug
        for reserved in ("CON", "con", "PRN", "AUX", "NUL", "COM1", "LPT9"):
            result = _slug(reserved)
            assert result.lower() not in {"con", "prn", "aux", "nul",
                                           "com1","com2","com3","com4","com5","com6","com7","com8","com9",
                                           "lpt1","lpt2","lpt3","lpt4","lpt5","lpt6","lpt7","lpt8","lpt9"}, \
                f"Reserved name {reserved!r} was not escaped — got {result!r}"

    def test_slug_path_traversal_neutralized(self):
        from history import _slug
        result = _slug("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result


# ---------------------------------------------------------------------------
# Handoff parser — empty next_headers fix
# ---------------------------------------------------------------------------

class TestHandoffParser:
    def test_last_section_extracted(self):
        from handoff_parser import _extract_section
        doc = "HEADER ONE\nfirst content\n\nHEADER TWO\nlast content here"
        result = _extract_section(doc, "HEADER TWO", next_headers=None)
        assert result == "last content here"

    def test_middle_section_extracted(self):
        from handoff_parser import _extract_section
        doc = "HEADER ONE\nfirst content\n\nHEADER TWO\nmiddle content\n\nHEADER THREE\nthird content"
        result = _extract_section(doc, "HEADER TWO", next_headers=["HEADER THREE"])
        assert "middle content" in result
        assert "third content" not in result
