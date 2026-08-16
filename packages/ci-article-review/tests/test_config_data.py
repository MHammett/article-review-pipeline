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

import pytest

import os

import yaml
import ci_article_review

_CONFIGS = os.path.join(os.path.dirname(ci_article_review.__file__), "configs")


def _load(name):
    with open(os.path.join(_CONFIGS, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestPresetsAreValid:
    """Validity, not parity (audit finding 14).

    This used to assert configs/presets.yaml matched a duplicate _COST_PRESETS
    dict in config_loader.py, so every model-name change had to be made twice.
    The duplicate is gone; the YAML is the single source of truth and the loader
    raises if it is missing or malformed. A parity test only proved two copies
    agreed, never that either was usable — these check that instead.
    """

    def test_every_documented_preset_exists(self):
        from ci_article_review.config_loader import _load_presets_from_yaml

        presets = _load_presets_from_yaml()
        expected = {"economy", "standard", "balanced", "thorough", "maximum"}
        assert expected <= set(presets), (
            f"presets.yaml is missing: {sorted(expected - set(presets))}. "
            "These are the choices offered by --cost-preset."
        )

    def test_each_preset_names_a_known_thoroughness(self):
        from ci_article_review.config_loader import _load_presets_from_yaml

        for name, body in _load_presets_from_yaml().items():
            assert body.get("thoroughness") in {"standard", "thorough", "maximum"}, (
                f"preset {name!r} has an unknown thoroughness "
                f"{body.get('thoroughness')!r}"
            )

    def test_each_preset_configures_only_real_providers(self):
        from ci_article_review.config_loader import _load_presets_from_yaml
        from ci_article_review.pipeline import _PROVIDERS

        for name, body in _load_presets_from_yaml().items():
            unknown = set(body.get("models", {})) - set(_PROVIDERS)
            assert not unknown, f"preset {name!r} names unknown providers: {unknown}"

    def test_a_missing_presets_file_raises_rather_than_degrading(self, tmp_path):
        import pytest

        from ci_core.config_helpers import PackagedConfigError
        from ci_article_review.config_loader import _load_presets_from_yaml

        with pytest.raises(PackagedConfigError):
            _load_presets_from_yaml(config_dir=tmp_path)


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


class TestPartialPipelineBlockKeepsDefaults:
    """A partial `pipeline:` block must not discard the keys it omits.

    The defaults used to be the fallback argument of
    `user_config.get("pipeline", {...})`, which applies only when the key is
    absent entirely. So a config as ordinary as

        pipeline:
          cost_preset: maximum

    silently dropped every default: `task_timeout_seconds` became None, which is
    a TypeError inside timeout_model.compute_timeout, and `retry_on_failure`
    became None, disabling retries without saying so.
    """

    def _pipeline(self, user):
        from ci_article_review.config_loader import merge_configs

        return merge_configs(user, {})["pipeline"]

    def test_absent_block_gets_every_default(self):
        from ci_article_review.config_loader import PIPELINE_DEFAULTS

        assert self._pipeline({}) == PIPELINE_DEFAULTS

    @pytest.mark.parametrize(
        "partial",
        [
            {"cost_preset": "maximum"},
            {"grammar_pass": False},
            {"link_validation": False},
        ],
    )
    def test_a_partial_block_keeps_the_keys_it_did_not_mention(self, partial):
        merged = self._pipeline({"pipeline": dict(partial)})
        assert merged["task_timeout_seconds"] == 180
        assert merged["retry_on_failure"] is True
        assert merged["retry_delay_seconds"] == 10
        # ...and still carries what the user did say.
        for key, value in partial.items():
            assert merged[key] == value

    def test_an_explicit_null_block_is_not_a_crash(self):
        """`pipeline:` with nothing under it parses as None, not {}."""
        assert self._pipeline({"pipeline": None})["task_timeout_seconds"] == 180

    def test_user_values_still_win(self):
        merged = self._pipeline(
            {"pipeline": {"task_timeout_seconds": 1100, "retry_on_failure": False}}
        )
        assert merged["task_timeout_seconds"] == 1100
        assert merged["retry_on_failure"] is False

    def test_the_computed_backstop_survives_a_partial_block(self):
        """The actual failure: None reached compute_timeout and raised TypeError."""
        from ci_core.llm import timeout_model

        merged = self._pipeline({"pipeline": {"cost_preset": "maximum"}})
        budget = timeout_model.compute_all(
            2054,
            {"openai": {"model": "gpt-5.5", "reasoning_effort": "xhigh"}},
            merged["task_timeout_seconds"],
        )["openai"]
        assert budget > 0
