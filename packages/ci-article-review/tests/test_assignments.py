"""Tests for pipeline._build_assignments — what it schedules, and what it drops.

Why this file exists
--------------------
A 2026-08-12 maximum-thoroughness run made 25 model calls where the preset asks
for 30 (6 models x 5 domains). Nothing in the run output said why. The cause was
``enabled: false`` on the claude entry in user.yaml — working as designed, but
finding that took two days and a direct call to ``_build_assignments``, because
the run logs the assignments it made and stays silent about the model it
dropped. The skip-reporting tests below pin the output that closes that gap.

The normalisation tests cover the other half of that debugging session: called
directly with the simple config form that user.example.yaml documents
(``openai: gpt-5.5``), the function used to raise ``AttributeError: 'str' object
has no attribute 'get'``. The real pipeline normalises via config_loader first,
so production never hit it — the contract just did not say so.
"""

import logging
import re

from contextlib import ExitStack
from unittest.mock import patch

import pytest

import ci_article_review.pipeline as pipeline
from ci_article_review.pipeline import _build_assignments


_ALL_MODELS = ["gemini", "openai", "mistral", "grok", "claude", "perplexity"]

#: Every model credentialled — so a test's skips come from what it configures.
_ALL_KEYS = {m: {"api_key": "k"} for m in _ALL_MODELS}

#: Simple form, as user.example.yaml ships it.
_SIMPLE_CONFIGS = {
    "openai": "gpt-5.4",
    "gemini": "gemini-2.5-flash",
    "mistral": "mistral-large-latest",
    "perplexity": "sonar-reasoning-pro",
    "grok": "grok-4.3",
    "claude": "claude-opus-4-8",
}


def _skip_for(skips, model_name):
    """Return the single skip line naming ``model_name``, asserting there is one."""
    matches = [s for s in skips if s.startswith(f"{model_name} ")]
    assert len(matches) == 1, (
        f"Expected exactly one skip line for {model_name!r}, got {matches!r}"
    )
    return matches[0]


class TestSkipReporting:
    """Every model the preset asks for but does not run explains itself."""

    def test_disabled_model_is_reported_with_its_reason(self):
        """The 2026-08-12 run, reproduced: 25 calls instead of 30, and why."""
        configs = dict(_SIMPLE_CONFIGS)
        configs["claude"] = {"enabled": False, "model": "claude-opus-4-8"}

        skips = []
        assignments = _build_assignments("maximum", configs, _ALL_KEYS, skips=skips)

        assert len(assignments) == 25
        assert not any(m == "claude" for m, _ in assignments)

        line = _skip_for(skips, "claude")
        assert "disabled" in line
        assert "enabled: false" in line
        # The five domains it would have run are named, not just counted.
        for domain in pipeline._DOMAIN_PROMPTS:
            assert domain in line, f"{domain!r} missing from skip line: {line}"

    def test_missing_credentials_are_reported_as_such(self):
        keys = {m: v for m, v in _ALL_KEYS.items() if m != "grok"}

        skips = []
        assignments = _build_assignments("maximum", _SIMPLE_CONFIGS, keys, skips=skips)

        assert len(assignments) == 25
        line = _skip_for(skips, "grok")
        assert "no credentials" in line
        assert "disabled" not in line

    def test_prompts_override_names_the_domains_it_excluded(self):
        configs = dict(_SIMPLE_CONFIGS)
        configs["claude"] = {
            "model": "claude-opus-4-8",
            "prompts": ["fact_check", "completeness"],
        }

        skips = []
        assignments = _build_assignments("maximum", configs, _ALL_KEYS, skips=skips)

        claude_domains = {d for m, d in assignments if m == "claude"}
        assert claude_domains == {"fact_check", "completeness"}

        line = _skip_for(skips, "claude")
        assert "prompts" in line
        # The three preset domains the override dropped, named.
        for domain in ("voice_style", "argument_integrity", "red_team"):
            assert domain in line, f"{domain!r} missing from skip line: {line}"
        # ...and not the two it kept, which appear only in the parenthetical
        # naming what the override limited it to.
        assert "not run: " in line
        not_run = line.split("not run: ", 1)[1].split(" (", 1)[0]
        assert "fact_check" not in not_run
        assert "completeness" not in not_run

    def test_one_line_per_skipped_model_not_per_domain(self):
        """A model dropped from five domains gets one line, not five."""
        configs = dict(_SIMPLE_CONFIGS)
        configs["claude"] = {"enabled": False, "model": "claude-opus-4-8"}
        configs["grok"] = {"enabled": False, "model": "grok-4.3"}
        keys = {m: v for m, v in _ALL_KEYS.items() if m != "perplexity"}

        skips = []
        _build_assignments("maximum", configs, keys, skips=skips)

        assert len(skips) == 3
        assert {s.split(" ", 1)[0] for s in skips} == {"claude", "grok", "perplexity"}

    def test_nothing_is_reported_when_the_full_preset_runs(self):
        """No false alarms: a complete ensemble reports no skips at all."""
        skips = []
        assignments = _build_assignments(
            "maximum", _SIMPLE_CONFIGS, _ALL_KEYS, skips=skips
        )

        assert len(assignments) == 30
        assert skips == []

    def test_a_disabled_model_without_credentials_reports_one_reason(self):
        """Both gates fail; the report stays one line and names the first."""
        configs = dict(_SIMPLE_CONFIGS)
        configs["claude"] = {"enabled": False, "model": "claude-opus-4-8"}
        keys = {m: v for m, v in _ALL_KEYS.items() if m != "claude"}

        skips = []
        _build_assignments("maximum", configs, keys, skips=skips)

        line = _skip_for(skips, "claude")
        assert "disabled" in line
        assert "no credentials" not in line

    def test_the_skipped_domains_reconcile_with_the_preset(self):
        """Assignments plus reported skips account for every preset slot.

        All four drop reasons at once, so the arithmetic has to hold across a
        model-level block, an override, and the drafter rule together.
        """
        configs = dict(_SIMPLE_CONFIGS)
        configs["claude"] = {"enabled": False, "model": "claude-opus-4-8"}
        configs["mistral"] = {
            "model": "mistral-large-latest",
            "prompts": ["red_team"],
        }
        keys = {m: v for m, v in _ALL_KEYS.items() if m != "grok"}

        skips = []
        assignments = _build_assignments(
            "maximum", configs, keys, "openai", skips=skips
        )

        preset_slots = sum(
            len(models) for models in pipeline._THOROUGHNESS_PRESETS["maximum"].values()
        )
        # Each line states its own count; that count is what has to reconcile.
        reported_skips = sum(
            int(re.search(r"(\d+) domain\(s\) not run", s).group(1)) for s in skips
        )
        assert len(assignments) + reported_skips == preset_slots

    def test_a_model_outside_the_preset_reports_its_own_prompts_entry(self):
        """standard has no perplexity slot; an explicit prompts: entry adds one."""
        configs = dict(_SIMPLE_CONFIGS)
        configs["perplexity"] = {
            "model": "sonar-reasoning-pro",
            "prompts": ["fact_check"],
        }
        keys = {m: v for m, v in _ALL_KEYS.items() if m != "perplexity"}

        skips = []
        assignments = _build_assignments("standard", configs, keys, skips=skips)

        assert ("perplexity", "fact_check") not in assignments
        line = _skip_for(skips, "perplexity")
        assert "no credentials" in line
        assert "fact_check" in line

    def test_an_unreported_model_is_one_the_preset_never_wanted(self):
        """standard asks for no perplexity, so a keyless perplexity is not a skip."""
        keys = {m: v for m, v in _ALL_KEYS.items() if m != "perplexity"}

        skips = []
        _build_assignments("standard", _SIMPLE_CONFIGS, keys, skips=skips)

        assert skips == []

    def test_the_drafting_model_exclusion_is_reported_too(self):
        """The quietest drop of the four: nothing in the config asks for it."""
        skips = []
        assignments = _build_assignments(
            "maximum", _SIMPLE_CONFIGS, _ALL_KEYS, "claude", skips=skips
        )

        assert ("claude", "voice_style") not in assignments
        assert len(assignments) == 29

        line = _skip_for(skips, "claude")
        assert "voice_style" in line
        assert "drafted this article" in line
        # The four domains it still reviews are not in the not-run list.
        for domain in ("fact_check", "completeness", "argument_integrity", "red_team"):
            assert domain not in line.split("not run: ", 1)[1]

    def test_a_model_dropped_two_ways_still_gets_one_line(self):
        """A prompts: override and the drafter rule, attributed separately."""
        configs = dict(_SIMPLE_CONFIGS)
        configs["claude"] = {
            "model": "claude-opus-4-8",
            "prompts": ["fact_check", "voice_style"],
        }

        skips = []
        _build_assignments("maximum", configs, _ALL_KEYS, "claude", skips=skips)

        line = _skip_for(skips, "claude")
        # voice_style survived the override, then the drafter rule took it.
        assert "drafted this article" in line
        assert "prompts: override" in line
        assert "4 domain(s) not run" in line

    def test_the_drafter_rule_is_not_reported_for_other_models(self):
        """openai drafted nothing here, so its voice_style pass is not a skip."""
        skips = []
        _build_assignments("maximum", _SIMPLE_CONFIGS, _ALL_KEYS, "claude", skips=skips)

        assert {s.split(" ", 1)[0] for s in skips} == {"claude"}

    def test_skips_argument_is_optional(self):
        """Existing callers that pass neither skips nor a drafter keep working."""
        configs = dict(_SIMPLE_CONFIGS)
        configs["claude"] = {"enabled": False, "model": "claude-opus-4-8"}

        assert len(_build_assignments("maximum", configs, _ALL_KEYS)) == 25

    def test_reporting_does_not_change_what_is_assigned(self):
        configs = dict(_SIMPLE_CONFIGS)
        configs["claude"] = {"model": "claude-opus-4-8", "prompts": ["fact_check"]}
        keys = {m: v for m, v in _ALL_KEYS.items() if m != "grok"}

        with_skips = _build_assignments("thorough", configs, keys, skips=[])
        without_skips = _build_assignments("thorough", configs, keys)
        assert with_skips == without_skips


class TestConfigFormNormalisation:
    """Both config forms user.example.yaml documents reach here intact."""

    def test_simple_string_form_does_not_raise(self):
        """Was: AttributeError: 'str' object has no attribute 'get'."""
        assignments = _build_assignments("maximum", _SIMPLE_CONFIGS, _ALL_KEYS)
        assert len(assignments) == 30

    def test_both_forms_agree(self):
        extended = {name: {"model": model} for name, model in _SIMPLE_CONFIGS.items()}
        assert _build_assignments("maximum", _SIMPLE_CONFIGS, _ALL_KEYS) == (
            _build_assignments("maximum", extended, _ALL_KEYS)
        )

    def test_forms_can_be_mixed(self):
        """user.example.yaml: 'Simple-form entries are unaffected — you can mix.'"""
        mixed = dict(_SIMPLE_CONFIGS)
        mixed["claude"] = {"enabled": False, "model": "claude-opus-4-8"}
        mixed["grok"] = {"model": "grok-4.3", "prompts": ["red_team"]}

        skips = []
        assignments = _build_assignments("maximum", mixed, _ALL_KEYS, skips=skips)

        assert {d for m, d in assignments if m == "grok"} == {"red_team"}
        assert not any(m == "claude" for m, _ in assignments)
        assert {s.split(" ", 1)[0] for s in skips} == {"claude", "grok"}

    def test_string_form_still_honours_a_missing_key(self):
        """Normalisation must not invent credentials."""
        keys = {m: v for m, v in _ALL_KEYS.items() if m != "mistral"}
        assignments = _build_assignments("maximum", _SIMPLE_CONFIGS, keys)
        assert not any(m == "mistral" for m, _ in assignments)

    def test_string_form_gemini_is_not_mistaken_for_vertex(self):
        """Normalisation fills in provider; the api-key path must still apply.

        _model_has_credentials treats provider: vertex_ai as needing a project
        rather than a key, so the default provider filled in here matters.
        """
        assignments = _build_assignments(
            "maximum", {"gemini": "gemini-2.5-flash"}, {"gemini": {"api_key": "k"}}
        )
        assert {d for m, d in assignments if m == "gemini"} == set(
            pipeline._DOMAIN_PROMPTS
        )

    def test_empty_prompts_override_runs_nothing_rather_than_raising(self):
        """`prompts:` with no value parses as None; it used to raise TypeError."""
        configs = dict(_SIMPLE_CONFIGS)
        configs["claude"] = {"model": "claude-opus-4-8", "prompts": None}

        skips = []
        assignments = _build_assignments("maximum", configs, _ALL_KEYS, skips=skips)

        assert not any(m == "claude" for m, _ in assignments)
        assert "prompts" in _skip_for(skips, "claude")


class TestSkipsReachTheRunOutput:
    """The reasons are only useful if the run actually prints them."""

    _CURRENCY = {
        "warnings": [],
        "notices": [],
        "registry_warning": False,
        "registry_stale": False,
        "registry_date": "",
        "registry_age_days": 0,
    }
    _HANDOFF = {
        "title": "A Title That Is Comfortably Long Enough",
        "draft": "# A Title That Is Comfortably Long Enough\n\n## Section\n\nBody.",
        "primary_claim": "The claim.",
        "run_number": 1,
    }

    def test_empty_ensemble_logs_why_before_it_exits(self, caplog):
        """The run that produces nothing is the one that most needs the reasons.

        _build_assignments is deliberately left unpatched here — this covers the
        wiring from config to log line, which patching it out would skip.
        """
        caplog.set_level(logging.INFO, logger="pipeline")

        config = {
            "api_keys": {"mistral": {"api_key": "k"}},
            "pipeline": {"link_validation": False, "grammar_pass": False},
            "publication": {},
            "delta": {},
            "ensemble": {},
            # Credentialled but switched off; every other model has no key.
            "models": {"mistral": {"enabled": False, "model": "mistral-large-latest"}},
        }

        with ExitStack() as stack:
            for target, kwargs in (
                ("load_user_config", {"return_value": {"pipeline": {}}}),
                ("load_publication_config", {"return_value": {}}),
                ("merge_configs", {"return_value": config}),
                ("check_model_currency", {"return_value": self._CURRENCY}),
                ("_build_custom_assignments", {"return_value": ([], {})}),
            ):
                stack.enter_context(
                    patch(f"ci_article_review.pipeline.{target}", **kwargs)
                )
            stack.enter_context(
                patch(
                    "ci_article_review.pipeline.seo_suggest.generate",
                    return_value=({"status": "skipped", "reason": "test"}, None),
                )
            )
            stack.enter_context(
                patch(
                    "ci_article_review.pipeline.seo_content.review",
                    return_value=({"status": "skipped", "reason": "test"}, None),
                )
            )
            with pytest.raises(SystemExit):
                pipeline.run_draft_pipeline(None, "myblog", handoff=dict(self._HANDOFF))

        skip_lines = [
            r.getMessage() for r in caplog.records if "Skipped:" in r.getMessage()
        ]
        # standard preset: mistral disabled, the other four have no credentials.
        assert len(skip_lines) == 5
        assert all(
            r.levelno == logging.INFO
            for r in caplog.records
            if "Skipped:" in r.getMessage()
        )

        mistral_line = next(s for s in skip_lines if "mistral" in s)
        assert "disabled" in mistral_line
        assert "no credentials" in next(s for s in skip_lines if "claude" in s)
