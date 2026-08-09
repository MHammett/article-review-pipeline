"""Tests for ci_core.config_helpers.

These helpers are the supported cross-package contract for YAML/env config
loading — ci-article-review and ci-style-profile both build their configs on
them, so a change here is a change to both callers.
"""

import os

import pytest


class TestConfigHelpers:
    def test_normalize_simple_string_gemini(self):
        from ci_core.config_helpers import normalize_model_configs

        result = normalize_model_configs({"gemini": "gemini-2.5-flash"})
        assert result["gemini"]["provider"] == "ai_studio"
        assert result["gemini"]["model"] == "gemini-2.5-flash"

    def test_normalize_simple_string_openai(self):
        from ci_core.config_helpers import normalize_model_configs

        result = normalize_model_configs({"openai": "gpt-4o"})
        assert result["openai"]["provider"] == "openai"
        assert result["openai"]["model"] == "gpt-4o"

    def test_normalize_extended_dict_passthrough(self):
        from ci_core.config_helpers import normalize_model_configs

        cfg = {
            "gemini": {
                "provider": "vertex_ai",
                "model": "gemini-2.5-flash",
                "project": "my-proj",
            }
        }
        result = normalize_model_configs(cfg)
        assert result["gemini"]["provider"] == "vertex_ai"
        assert result["gemini"]["project"] == "my-proj"

    def test_normalize_dict_without_provider_gets_default(self):
        from ci_core.config_helpers import normalize_model_configs

        result = normalize_model_configs({"mistral": {"model": "mistral-large-latest"}})
        assert result["mistral"]["provider"] == "mistral"

    def test_normalize_mixed_forms(self):
        from ci_core.config_helpers import normalize_model_configs

        raw = {
            "openai": "gpt-4o",
            "gemini": {
                "provider": "vertex_ai",
                "model": "gemini-2.5-flash",
                "project": "p",
            },
        }
        result = normalize_model_configs(raw)
        assert result["openai"]["provider"] == "openai"
        assert result["gemini"]["provider"] == "vertex_ai"

    def test_normalize_simple_string_claude(self):
        from ci_core.config_helpers import normalize_model_configs

        result = normalize_model_configs({"claude": "claude-opus-4-5"})
        assert result["claude"]["provider"] == "anthropic"
        assert result["claude"]["model"] == "claude-opus-4-5"

    def test_normalize_preserves_enabled_flag(self):
        from ci_core.config_helpers import normalize_model_configs

        result = normalize_model_configs(
            {"grok": {"model": "grok-3-latest", "enabled": False}}
        )
        assert result["grok"]["enabled"] is False
        assert result["grok"]["provider"] == "grok"

    def test_normalize_enabled_defaults_absent_for_simple_form(self):
        from ci_core.config_helpers import normalize_model_configs

        result = normalize_model_configs({"openai": "gpt-4o"})
        # Simple form has no enabled key — caller should default to True
        assert "enabled" not in result["openai"]

    def test_normalize_empty_input(self):
        from ci_core.config_helpers import normalize_model_configs

        assert normalize_model_configs({}) == {}
        assert normalize_model_configs(None) == {}

    def test_env_var_missing_gives_helpful_error(self):
        from ci_core.config_helpers import resolve_env

        env_key = "PIPELINE_TEST_MISSING_VAR_XYZ"
        if env_key in os.environ:
            del os.environ[env_key]
        with pytest.raises(ValueError, match="not set"):
            resolve_env(f"${{{env_key}}}")
