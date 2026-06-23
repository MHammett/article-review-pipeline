"""Tests for model_registry.check_model_currency."""
import sys
import os
import datetime
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_registry import check_model_currency, REGISTRY_DATE, _SUPERSEDED, _NEWER_AVAILABLE


class TestCheckModelCurrency:
    def test_current_models_no_warnings(self):
        models = {
            "openai":     {"model": "gpt-5.4",            "provider": "openai"},
            "gemini":     {"model": "gemini-2.5-flash",   "provider": "ai_studio"},
            "grok":       {"model": "grok-4.3",           "provider": "grok"},
            "claude":     {"model": "claude-opus-4-8",    "provider": "anthropic"},
            "mistral":    {"model": "mistral-large-latest","provider": "mistral"},
            "perplexity": {"model": "sonar-reasoning-pro","provider": "perplexity"},
        }
        result = check_model_currency(models)
        assert result["warnings"] == [], f"unexpected warnings: {result['warnings']}"

    def test_superseded_model_triggers_warning(self):
        models = {
            "openai": {"model": "gpt-4o", "provider": "openai"},
        }
        result = check_model_currency(models)
        assert len(result["warnings"]) == 1
        w = result["warnings"][0]
        assert w["provider"] == "openai"
        assert w["model"] == "gpt-4o"
        assert w["replacement"] == "gpt-5.4"

    def test_multiple_superseded(self):
        models = {
            "openai": {"model": "gpt-4o",         "provider": "openai"},
            "grok":   {"model": "grok-3-latest",  "provider": "grok"},
            "claude": {"model": "claude-opus-4-5","provider": "anthropic"},
        }
        result = check_model_currency(models)
        assert len(result["warnings"]) == 3

    def test_disabled_model_skipped(self):
        models = {
            "openai": {"model": "gpt-4o", "provider": "openai", "enabled": False},
        }
        result = check_model_currency(models)
        assert result["warnings"] == []

    def test_newer_available_is_notice_not_warning(self):
        models = {
            "openai": {"model": "gpt-5.4", "provider": "openai"},
        }
        result = check_model_currency(models)
        assert result["warnings"] == []
        assert len(result["notices"]) == 1
        assert result["notices"][0]["newer"] == "gpt-5.5"

    def test_empty_config_no_crash(self):
        result = check_model_currency({})
        assert result["warnings"] == []
        assert result["notices"] == []

    def test_none_config_no_crash(self):
        result = check_model_currency(None)
        assert result["warnings"] == []

    def test_registry_age_fields_present(self):
        result = check_model_currency({})
        assert "registry_date" in result
        assert "registry_age_days" in result
        assert isinstance(result["registry_age_days"], int)
        assert result["registry_age_days"] >= 0

    def test_registry_date_is_valid_iso(self):
        result = check_model_currency({})
        parsed = datetime.date.fromisoformat(result["registry_date"])
        assert parsed == REGISTRY_DATE

    def test_all_superseded_keys_are_strings(self):
        for k in _SUPERSEDED:
            assert isinstance(k, str), f"superseded key {k!r} is not a string"

    def test_all_superseded_have_replacement(self):
        for model_id, info in _SUPERSEDED.items():
            assert "replacement" in info, f"{model_id!r} missing 'replacement'"
            assert isinstance(info["replacement"], str), f"{model_id!r} replacement is not a string"
