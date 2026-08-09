"""Integrity tests for ci-article-review's externalized config data.

  configs/presets.yaml -> config_loader.py (_COST_PRESETS)

The loader keeps a hardcoded fallback used only when the YAML is missing or
malformed.  These tests assert two things:

1. PARITY — the YAML content matches the hardcoded fallback, so a dropped or
   corrupt YAML silently using stale defaults can never go unnoticed.  If you
   intentionally change one, change the other and these tests confirm they agree.
2. STRUCTURE — the YAML parses and has the shape the loader expects, so a typo
   is caught at test time instead of during a real (billed) pipeline run.

The pricing / model_registry / timeouts data files moved to ci-core alongside
their loaders; their equivalents live in ci-core/tests/test_llm_config_data.py.
"""

import os

import yaml
import ci_article_review

_CONFIGS = os.path.join(os.path.dirname(ci_article_review.__file__), "configs")


def _load(name):
    with open(os.path.join(_CONFIGS, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestPresetParity:
    def test_loaded_presets_match_fallback(self):
        from ci_article_review.config_loader import (
            _load_presets_from_yaml,
            _COST_PRESETS,
        )

        loaded = _load_presets_from_yaml()
        assert loaded is not None, "configs/presets.yaml failed to load"
        assert loaded == _COST_PRESETS, (
            "configs/presets.yaml has drifted from _COST_PRESETS in config_loader.py — "
            "update both or a missing YAML will silently run stale preset models."
        )


class TestPresetsStructure:
    _EXPECTED = {"economy", "standard", "balanced", "thorough", "maximum"}
    _VALID_THOROUGHNESS = {"standard", "thorough", "maximum"}

    def test_all_five_presets_present(self):
        data = _load("presets.yaml")
        assert self._EXPECTED.issubset(set(data.keys()))

    def test_each_preset_has_valid_thoroughness_and_models(self):
        data = _load("presets.yaml")
        for name in self._EXPECTED:
            body = data[name]
            assert body["thoroughness"] in self._VALID_THOROUGHNESS, (
                f"{name}: bad thoroughness"
            )
            assert isinstance(body["models"], dict) and body["models"], (
                f"{name}: no models"
            )

    def test_no_grok_reasoning_effort(self):
        # Grok reasoning is model-selection based; reasoning_effort is not a valid Grok param.
        data = _load("presets.yaml")
        for name, body in data.items():
            grok = body.get("models", {}).get("grok") or {}
            assert "reasoning_effort" not in grok, (
                f"{name}: grok has reasoning_effort — use grok-4.20-0309-reasoning instead"
            )

    def test_mistral_reasoning_effort_is_high_or_none_only(self):
        # mistral-medium-3-5 only accepts "high" or "none"; "low"/"medium" return 400.
        data = _load("presets.yaml")
        for name, body in data.items():
            mistral = body.get("models", {}).get("mistral") or {}
            effort = mistral.get("reasoning_effort")
            if effort is not None:
                assert effort in ("high", "none"), (
                    f"{name}: mistral reasoning_effort={effort!r} invalid"
                )
