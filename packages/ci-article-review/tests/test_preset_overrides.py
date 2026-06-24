"""Tests for config_loader._apply_preset_overrides and cost preset interaction."""

import pytest


from ci_article_review.config_loader import _apply_preset_overrides, _apply_cost_preset


class TestApplyPresetOverrides:
    def test_no_overrides_returns_models_unchanged(self):
        models = {"openai": {"model": "gpt-5.4"}}
        result = _apply_preset_overrides({"cost_preset": "balanced"}, models)
        assert result["openai"]["model"] == "gpt-5.4"

    def test_override_single_key(self):
        models = {"openai": {"model": "gpt-5.4", "reasoning_effort": "low"}}
        pipeline = {
            "cost_preset": "balanced",
            "preset_overrides": {"openai": {"reasoning_effort": "high"}},
        }
        result = _apply_preset_overrides(pipeline, models)
        assert result["openai"]["reasoning_effort"] == "high"
        assert result["openai"]["model"] == "gpt-5.4"  # unchanged

    def test_override_model_and_effort(self):
        # Overrides add/change keys; they do NOT remove keys not mentioned.
        # To neutralize thinking_budget, set it explicitly to null.
        models = {"claude": {"model": "claude-sonnet-4-6", "thinking_budget": 8000}}
        pipeline = {
            "preset_overrides": {
                "claude": {"model": "claude-opus-4-8", "effort": "high"},
            }
        }
        result = _apply_preset_overrides(pipeline, models)
        assert result["claude"]["model"] == "claude-opus-4-8"
        assert result["claude"]["effort"] == "high"
        assert (
            result["claude"]["thinking_budget"] == 8000
        )  # still present; set to null to neutralize

    def test_override_null_removes_effect_of_key(self):
        # Setting a key to None (YAML: null) effectively disables it —
        # all adapters guard with "if cfg.get('key'):" so None == disabled.
        models = {"claude": {"model": "claude-sonnet-4-6", "thinking_budget": 8000}}
        pipeline = {
            "preset_overrides": {
                "claude": {
                    "model": "claude-opus-4-8",
                    "effort": "high",
                    "thinking_budget": None,
                },
            }
        }
        result = _apply_preset_overrides(pipeline, models)
        assert result["claude"]["model"] == "claude-opus-4-8"
        assert result["claude"]["thinking_budget"] is None  # disabled via null

    def test_override_ignores_unconfigured_provider(self):
        models = {"openai": {"model": "gpt-5.4"}}
        pipeline = {
            "preset_overrides": {
                "claude": {"model": "claude-opus-4-8"},  # claude not in models
            }
        }
        result = _apply_preset_overrides(pipeline, models)
        assert "claude" not in result

    def test_override_preserves_other_providers(self):
        models = {
            "openai": {"model": "gpt-5.4"},
            "mistral": {"model": "mistral-large-latest"},
        }
        pipeline = {"preset_overrides": {"openai": {"reasoning_effort": "xhigh"}}}
        result = _apply_preset_overrides(pipeline, models)
        assert result["mistral"]["model"] == "mistral-large-latest"

    def test_override_can_disable_provider(self):
        models = {"grok": {"model": "grok-4.3"}}
        pipeline = {"preset_overrides": {"grok": {"enabled": False}}}
        result = _apply_preset_overrides(pipeline, models)
        assert result["grok"]["enabled"] is False

    def test_none_overrides_is_noop(self):
        models = {"openai": {"model": "gpt-5.4"}}
        result = _apply_preset_overrides({"cost_preset": "balanced"}, models)
        assert result == models

    def test_empty_overrides_is_noop(self):
        models = {"openai": {"model": "gpt-5.4"}}
        result = _apply_preset_overrides({"preset_overrides": {}}, models)
        assert result == models

    def test_override_on_top_of_preset(self):
        """Full flow: preset sets balanced, override bumps openai to high reasoning."""
        models_raw = {
            "openai": "gpt-5.4",
            "gemini": "gemini-2.5-flash",
            "mistral": "mistral-large-latest",
        }
        pipeline = {
            "cost_preset": "balanced",
            "preset_overrides": {
                "openai": {"reasoning_effort": "high"},
            },
        }
        pipeline, models_after_preset = _apply_cost_preset(pipeline, models_raw)
        models_final = _apply_preset_overrides(pipeline, models_after_preset)

        # Preset set low; override should have bumped it to high
        assert models_final["openai"]["reasoning_effort"] == "high"
        # Preset's model selection should still be in effect
        assert models_final["openai"]["model"] == "gpt-5.4"

    def test_invalid_overrides_type_raises(self):
        with pytest.raises(ValueError, match="preset_overrides must be a mapping"):
            _apply_preset_overrides({"preset_overrides": "not-a-dict"}, {})
