"""Scoping of the OpenAI ``web_search`` flag to the domains that can use it.

``web_search`` is a per-model flag in the adapters, so enabling it applied live
search to every domain that model ran. At ``maximum`` thoroughness OpenAI runs
all five, and search bills per search on top of tokens — so four of the five
were paying for a live page fetch to answer a question about the draft already
in the prompt. Only ``fact_check`` has any use for it, where a live-fetched
``source`` replaces training recall.

These tests pin the resolution rules and the wiring that applies them.
"""

from unittest.mock import patch

import pytest

from ci_article_review.pipeline import _run_domain, _web_search_enabled


class TestWebSearchEnabled:
    """Resolution of the ``web_search`` setting for a single domain."""

    def test_true_still_means_every_domain(self):
        """Existing configs that predate scoping keep their behaviour."""
        for domain in (
            "fact_check",
            "voice_style",
            "completeness",
            "argument_integrity",
            "red_team",
        ):
            assert _web_search_enabled(True, domain) is True

    def test_list_enables_only_listed_domains(self):
        assert _web_search_enabled(["fact_check"], "fact_check") is True
        assert _web_search_enabled(["fact_check"], "voice_style") is False

    def test_list_may_name_several_domains(self):
        setting = ["fact_check", "red_team"]
        assert _web_search_enabled(setting, "fact_check") is True
        assert _web_search_enabled(setting, "red_team") is True
        assert _web_search_enabled(setting, "completeness") is False

    def test_bare_string_is_treated_as_one_domain(self):
        """`web_search: fact_check` is a domain name, not a truthy value.

        Falling through to ``bool("fact_check")`` would enable search on all
        five domains — the exact bill the list form exists to prevent.
        """
        assert _web_search_enabled("fact_check", "fact_check") is True
        assert _web_search_enabled("fact_check", "voice_style") is False

    def test_absent_or_false_disables(self):
        assert _web_search_enabled(None, "fact_check") is False
        assert _web_search_enabled(False, "fact_check") is False

    def test_unknown_domain_name_fails_closed(self):
        """A typo costs coverage, not money."""
        assert _web_search_enabled(["fact-check"], "fact_check") is False
        assert _web_search_enabled([], "fact_check") is False


class TestRunDomainAppliesTheScope:
    """The gate is applied where the domain is known, not in the adapter."""

    @staticmethod
    def _provider_config_for(domain, web_search_setting):
        """Run one domain and return the provider_config the LLM layer received."""
        captured = {}

        def _fake_call(provider, system, user, api_key, **kwargs):
            captured.update(kwargs["provider_config"])
            return {"raw": "{}"}

        model_configs = {"openai": {"model": "gpt-5.4"}}
        if web_search_setting is not None:
            model_configs["openai"]["web_search"] = web_search_setting

        with patch(
            "ci_article_review.pipeline.llm.call_provider",
            side_effect=_fake_call,
        ):
            _run_domain(
                "openai",
                domain,
                "draft body",
                {"title": "T"},
                {},
                {"openai": {"api_key": "k"}},
                {},
                model_configs,
            )
        return captured

    @pytest.mark.parametrize(
        "domain,expected",
        [
            ("fact_check", True),
            ("voice_style", False),
            ("completeness", False),
            ("argument_integrity", False),
            ("red_team", False),
        ],
    )
    def test_list_form_reaches_the_adapter_per_domain(self, domain, expected):
        cfg = self._provider_config_for(domain, ["fact_check"])
        assert cfg["web_search"] is expected

    def test_absent_flag_is_not_introduced(self):
        """Models that never configured web_search keep a config without it.

        The key is resolved in place rather than added, so a provider whose
        adapter does something else with unknown keys is unaffected.
        """
        cfg = self._provider_config_for("fact_check", None)
        assert "web_search" not in cfg
