"""Tests for redact — secret scrubbing before error output."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from redact import redact_url_keys, redact_value


class TestRedactUrlKeys:
    def test_redacts_gemini_key_query_param(self):
        err = ("HTTPSConnectionPool(host='generativelanguage.googleapis.com', "
               "port=443): url: /v1beta/models?key=AIzaSyABC_REALKEY123 (Caused by ...)")
        out = redact_url_keys(err)
        assert "AIzaSyABC_REALKEY123" not in out
        assert "key=[REDACTED]" in out

    def test_redacts_apikey_variants(self):
        for param in ("apiKey", "api_key", "access_token", "token"):
            err = f"https://api.example.com/x?{param}=SECRETVALUE&foo=bar"
            out = redact_url_keys(err)
            assert "SECRETVALUE" not in out, f"{param} not redacted"
            assert "foo=bar" in out, "should not eat following params"

    def test_preserves_non_key_text(self):
        msg = "Connection timed out after 20s"
        assert redact_url_keys(msg) == msg

    def test_accepts_exception_object(self):
        exc = Exception("failed: https://x.com/m?key=ABC123")
        out = redact_url_keys(exc)
        assert "ABC123" not in out

    def test_case_insensitive_param_name(self):
        err = "https://x.com/m?KEY=SECRET"
        assert "SECRET" not in redact_url_keys(err)


class TestRedactValue:
    def test_redacts_known_value(self):
        assert redact_value("error: sk-abc123 failed", "sk-abc123") == "error: [REDACTED] failed"

    def test_no_op_when_value_absent(self):
        assert redact_value("clean message", "sk-abc123") == "clean message"

    def test_empty_secret_is_noop(self):
        assert redact_value("anything", "") == "anything"
