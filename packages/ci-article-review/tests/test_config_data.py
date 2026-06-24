"""Integrity tests for the externalized config data files.

Three data files were moved out of Python and into YAML:
  configs/pricing.yaml        -> analysis/cost.py
  configs/model_registry.yaml -> model_registry.py
  configs/presets.yaml        -> config_loader.py

Each loader keeps a hardcoded fallback used only when the YAML is missing or
malformed.  These tests assert two things:

1. PARITY — the YAML content matches the hardcoded fallback, so a dropped or
   corrupt YAML silently using stale defaults can never go unnoticed.  If you
   intentionally change one, change the other and these tests confirm they agree.
2. STRUCTURE — each YAML file parses and has the shape the loader expects, so a
   typo is caught at test time instead of during a real (billed) pipeline run.
"""

import os
import datetime

import yaml
import ci_article_review

_CONFIGS = os.path.join(os.path.dirname(ci_article_review.__file__), "configs")


def _load(name):
    with open(os.path.join(_CONFIGS, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Parity: YAML (as loaded by the module) == hardcoded fallback
# ---------------------------------------------------------------------------


class TestPricingParity:
    def test_loaded_pricing_matches_fallback(self):
        from ci_article_review.analysis import cost

        assert cost._PRICING == cost._PRICING_FALLBACK, (
            "configs/pricing.yaml has drifted from _PRICING_FALLBACK in analysis/cost.py — "
            "update both or the fallback will serve stale prices when the YAML is missing."
        )

    def test_loaded_unknown_price_matches_fallback(self):
        from ci_article_review.analysis import cost

        assert cost._UNKNOWN_PRICE == cost._UNKNOWN_PRICE_FALLBACK


class TestRegistryParity:
    def test_superseded_matches_fallback(self):
        import ci_article_review.model_registry as mr

        assert mr._SUPERSEDED == mr._SUPERSEDED_FALLBACK, (
            "configs/model_registry.yaml superseded: has drifted from "
            "_SUPERSEDED_FALLBACK in model_registry.py."
        )

    def test_newer_available_matches_fallback(self):
        import ci_article_review.model_registry as mr

        assert mr._NEWER_AVAILABLE == mr._NEWER_AVAILABLE_FALLBACK


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


class TestTimeoutParity:
    def test_loaded_timeouts_match_fallback(self):
        import ci_article_review.timeout_model as tm

        assert tm._CONFIG == tm._FALLBACK, (
            "configs/timeouts.yaml has drifted from _FALLBACK in timeout_model.py — "
            "update both or a missing YAML will silently use stale timeout multipliers."
        )


# ---------------------------------------------------------------------------
# Structure: each YAML parses and matches the loader's expectations
# ---------------------------------------------------------------------------


class TestPricingStructure:
    def test_every_model_has_numeric_pair(self):
        data = _load("pricing.yaml")
        for model_id, pair in data["models"].items():
            assert isinstance(pair, list) and len(pair) == 2, (
                f"{model_id}: not a 2-element list"
            )
            assert all(isinstance(x, (int, float)) for x in pair), (
                f"{model_id}: non-numeric price"
            )
            assert all(x >= 0 for x in pair), f"{model_id}: negative price"

    def test_unknown_price_present_and_numeric(self):
        data = _load("pricing.yaml")
        pair = data["unknown_price"]
        assert len(pair) == 2 and all(isinstance(x, (int, float)) for x in pair)


class TestRegistryStructure:
    def test_registry_date_is_valid_iso(self):
        data = _load("model_registry.yaml")
        # fromisoformat raises if malformed; str() handles a YAML-parsed date object too.
        datetime.date.fromisoformat(str(data["registry_date"]))

    def test_staleness_thresholds_are_ints(self):
        data = _load("model_registry.yaml")
        assert isinstance(data["stale_notice_days"], int)
        assert isinstance(data["stale_warning_days"], int)
        assert data["stale_notice_days"] < data["stale_warning_days"]

    def test_every_superseded_has_replacement(self):
        data = _load("model_registry.yaml")
        for model_id, info in data["superseded"].items():
            assert "replacement" in info, f"{model_id}: missing replacement"
            assert isinstance(info["replacement"], str)

    def test_every_newer_available_has_newer(self):
        data = _load("model_registry.yaml")
        for model_id, info in (data.get("newer_available") or {}).items():
            assert "newer" in info, f"{model_id}: missing 'newer'"


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


class TestTimeoutsStructure:
    def test_required_sections_present(self):
        data = _load("timeouts.yaml")
        for key in (
            "base_seconds",
            "floor_seconds",
            "size_multipliers",
            "model_multipliers",
            "effort_multipliers",
        ):
            assert key in data, f"timeouts.yaml missing '{key}'"

    def test_size_buckets_ordered_and_open_ended(self):
        buckets = _load("timeouts.yaml")["size_multipliers"]
        caps = [b["max_chars"] for b in buckets]
        assert caps[-1] is None, (
            "largest size bucket must be open-ended (max_chars: null)"
        )
        finite = [c for c in caps if c is not None]
        assert finite == sorted(finite), "size buckets must be in ascending order"

    def test_effort_table_has_default(self):
        eff = _load("timeouts.yaml")["effort_multipliers"]
        assert "default" in eff, (
            "effort_multipliers needs a 'default' for models with no effort param"
        )

    def test_model_table_has_default(self):
        models = _load("timeouts.yaml")["model_multipliers"]
        assert "default" in models, "model_multipliers needs a 'default'"

    def test_effort_is_monotonic_nondecreasing(self):
        eff = _load("timeouts.yaml")["effort_multipliers"]
        ladder = [
            eff[k] for k in ("none", "low", "medium", "high", "xhigh") if k in eff
        ]
        assert ladder == sorted(ladder), (
            f"effort multipliers should not decrease: {ladder}"
        )

    def test_variance_margin_present_and_sane(self):
        vm = _load("timeouts.yaml").get("variance_margin")
        assert vm is not None, "timeouts.yaml should define variance_margin"
        assert vm >= 1.0, (
            "variance_margin below 1.0 would shrink timeouts below the central estimate"
        )
