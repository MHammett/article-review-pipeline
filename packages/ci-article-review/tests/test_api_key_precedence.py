"""Tests for the four-tier API key precedence: CLI override > publication
config file > user.yaml/.env > OS environment variable.

Built after a real incident where a stale OS-level OPENAI_API_KEY silently
overrode .env with no way to override it back short of editing .env or
unsetting the shell variable. The fix has three independent pieces, each
covered here: user.yaml/.env's ${VAR} resolution now prefers a .env-defined
value over a same-named OS variable (see ci_core.env_provenance and
test_env_provenance.py / test_config_helpers.py for that layer specifically),
a publication config can override specific providers' keys, and --api-key on
the CLI overrides everything for a single run.
"""

import pytest

import ci_article_review.config_loader as config_loader
from ci_article_review.config_loader import (
    apply_api_key_overrides,
    apply_wordpress_overrides,
    load_publication_config,
    load_user_config,
    merge_configs,
    parse_api_key_overrides,
)


class TestParseApiKeyOverrides:
    def test_none_or_empty_yields_no_overrides(self):
        assert parse_api_key_overrides(None) == {}
        assert parse_api_key_overrides([]) == {}

    def test_parses_provider_equals_value_as_the_api_key_field(self):
        assert parse_api_key_overrides(["openai=sk-abc123"]) == {
            ("openai", "api_key"): "sk-abc123"
        }

    def test_parses_multiple_entries(self):
        result = parse_api_key_overrides(["openai=sk-abc", "claude=sk-ant-xyz"])
        assert result == {
            ("openai", "api_key"): "sk-abc",
            ("claude", "api_key"): "sk-ant-xyz",
        }

    def test_a_value_containing_equals_signs_is_preserved(self):
        """API keys can contain '=' (e.g. base64-ish tokens) — split on the
        first '=' only."""
        result = parse_api_key_overrides(["gemini=AQ.Ab8==tail"])
        assert result == {("gemini", "api_key"): "AQ.Ab8==tail"}

    def test_explicit_field_form_for_a_multi_field_credential(self):
        result = parse_api_key_overrides(
            ["languagetool.username=me@example.com", "languagetool.api_key=xyz"]
        )
        assert result == {
            ("languagetool", "username"): "me@example.com",
            ("languagetool", "api_key"): "xyz",
        }

    def test_archive_org_access_and_secret_key(self):
        result = parse_api_key_overrides(
            ["archive_org.access_key=AK", "archive_org.secret_key=SK"]
        )
        assert result == {
            ("archive_org", "access_key"): "AK",
            ("archive_org", "secret_key"): "SK",
        }

    def test_missing_equals_sign_raises(self):
        with pytest.raises(ValueError, match="PROVIDER=VALUE"):
            parse_api_key_overrides(["openai-sk-abc123"])

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="PROVIDER=VALUE"):
            parse_api_key_overrides(["=sk-abc123"])

    def test_unknown_provider_name_raises(self):
        with pytest.raises(ValueError, match="not-a-real-provider"):
            parse_api_key_overrides(["not-a-real-provider=value"])

    def test_unsupported_field_for_a_single_field_provider_raises(self):
        with pytest.raises(ValueError, match="openai"):
            parse_api_key_overrides(["openai.organization=org-123"])

    def test_unsupported_field_for_a_multi_field_provider_raises(self):
        with pytest.raises(ValueError, match="languagetool"):
            parse_api_key_overrides(["languagetool.password=x"])


class TestApplyApiKeyOverrides:
    def test_no_overrides_returns_the_same_config(self):
        config = {"api_keys": {"openai": {"api_key": "original"}}}
        assert apply_api_key_overrides(config, {}) is config

    def test_override_replaces_just_the_api_key_field(self):
        config = {
            "api_keys": {"openai": {"api_key": "original", "organization": "org-123"}}
        }
        result = apply_api_key_overrides(config, {("openai", "api_key"): "overridden"})
        assert result["api_keys"]["openai"] == {
            "api_key": "overridden",
            "organization": "org-123",
        }

    def test_override_does_not_touch_other_providers(self):
        config = {
            "api_keys": {
                "openai": {"api_key": "openai_original"},
                "claude": {"api_key": "claude_original"},
            }
        }
        result = apply_api_key_overrides(config, {("openai", "api_key"): "overridden"})
        assert result["api_keys"]["claude"]["api_key"] == "claude_original"

    def test_original_config_is_not_mutated(self):
        config = {"api_keys": {"openai": {"api_key": "original"}}}
        apply_api_key_overrides(config, {("openai", "api_key"): "overridden"})
        assert config["api_keys"]["openai"]["api_key"] == "original"

    def test_can_introduce_a_provider_not_previously_configured(self):
        config = {"api_keys": {}}
        result = apply_api_key_overrides(config, {("grok", "api_key"): "new_value"})
        assert result["api_keys"]["grok"]["api_key"] == "new_value"

    def test_multi_field_credential_sets_only_the_named_field(self):
        config = {
            "api_keys": {
                "languagetool": {"username": "original_user", "api_key": "original_key"}
            }
        }
        result = apply_api_key_overrides(
            config, {("languagetool", "api_key"): "new_key"}
        )
        assert result["api_keys"]["languagetool"] == {
            "username": "original_user",
            "api_key": "new_key",
        }


class TestApplyWordpressOverrides:
    """--wp-user / --wp-password: WordPress credentials live in
    publication.wordpress, not api_keys, so they get their own apply
    function rather than going through apply_api_key_overrides."""

    def test_no_overrides_returns_the_same_config(self):
        config = {"publication": {"wordpress": {"username": "original"}}}
        assert apply_wordpress_overrides(config) is config

    def test_username_override_leaves_password_untouched(self):
        config = {
            "publication": {
                "wordpress": {
                    "username": "original_user",
                    "application_password": "original_pw",
                }
            }
        }
        result = apply_wordpress_overrides(config, username="new_user")
        assert result["publication"]["wordpress"] == {
            "username": "new_user",
            "application_password": "original_pw",
        }

    def test_password_override_leaves_username_untouched(self):
        config = {
            "publication": {
                "wordpress": {
                    "username": "original_user",
                    "application_password": "original_pw",
                }
            }
        }
        result = apply_wordpress_overrides(config, application_password="new_pw")
        assert result["publication"]["wordpress"] == {
            "username": "original_user",
            "application_password": "new_pw",
        }

    def test_both_overridden_together(self):
        config = {
            "publication": {"wordpress": {"username": "u", "application_password": "p"}}
        }
        result = apply_wordpress_overrides(
            config, username="new_u", application_password="new_p"
        )
        assert result["publication"]["wordpress"] == {
            "username": "new_u",
            "application_password": "new_p",
        }

    def test_other_publication_fields_are_preserved(self):
        config = {
            "publication": {
                "wordpress": {"username": "u"},
                "style_profile": "some style notes",
            }
        }
        result = apply_wordpress_overrides(config, username="new_u")
        assert result["publication"]["style_profile"] == "some style notes"

    def test_original_config_is_not_mutated(self):
        config = {"publication": {"wordpress": {"username": "original"}}}
        apply_wordpress_overrides(config, username="new_user")
        assert config["publication"]["wordpress"]["username"] == "original"


class TestPublicationLevelApiKeyOverride:
    """merge_configs: a publication config's api_keys section overrides
    user.yaml's, per provider/field — the second precedence tier."""

    def test_publication_overrides_one_provider(self):
        user_config = {
            "api_keys": {
                "openai": {"api_key": "user_openai_key"},
                "claude": {"api_key": "user_claude_key"},
            }
        }
        pub_config = {"api_keys": {"openai": {"api_key": "pub_openai_key"}}}
        merged = merge_configs(user_config, pub_config)
        assert merged["api_keys"]["openai"]["api_key"] == "pub_openai_key"
        assert merged["api_keys"]["claude"]["api_key"] == "user_claude_key"

    def test_publication_can_add_a_field_without_dropping_others(self):
        user_config = {
            "api_keys": {"openai": {"api_key": "user_key", "organization": "org-1"}}
        }
        pub_config = {"api_keys": {"openai": {"organization": "org-2"}}}
        merged = merge_configs(user_config, pub_config)
        assert merged["api_keys"]["openai"] == {
            "api_key": "user_key",
            "organization": "org-2",
        }

    def test_no_publication_api_keys_leaves_user_config_untouched(self):
        user_config = {"api_keys": {"openai": {"api_key": "user_key"}}}
        merged = merge_configs(user_config, {})
        assert merged["api_keys"]["openai"]["api_key"] == "user_key"

    def test_publication_can_introduce_a_provider_absent_from_user_yaml(self):
        user_config = {"api_keys": {"openai": {"api_key": "user_key"}}}
        pub_config = {"api_keys": {"perplexity": {"api_key": "pub_only_key"}}}
        merged = merge_configs(user_config, pub_config)
        assert merged["api_keys"]["perplexity"]["api_key"] == "pub_only_key"
        assert merged["api_keys"]["openai"]["api_key"] == "user_key"


class TestFullPrecedenceChain:
    """End to end, through the real file-loading path: CLI > publication
    config > user.yaml/.env > OS environment variable."""

    @pytest.fixture
    def config_dir(self, tmp_path, monkeypatch):
        # Patch the module's frozen snapshot directly rather than
        # monkeypatch.setenv — _EFFECTIVE_ENV is built once at import time
        # from whatever real .env exists on the machine running the suite,
        # and (correctly, per the precedence this module enforces) a
        # .env-defined value would keep beating a monkeypatched OS variable
        # of the same name. Patching it keeps this test independent of the
        # real environment, same reasoning as test_key_sources.py's
        # explicit env_snapshot passing.
        monkeypatch.setattr(
            config_loader,
            "_EFFECTIVE_ENV",
            {
                "OPENAI_API_KEY": "env_tier_key",
                "MISTRAL_API_KEY": "env_tier_mistral_key",
            },
        )
        (tmp_path / "user.yaml").write_text(
            """
api_keys:
  openai:
    api_key: ${OPENAI_API_KEY}
  gemini:
    api_key: dummy_gemini_key
  mistral:
    api_key: ${MISTRAL_API_KEY}
""",
            encoding="utf-8",
        )
        (tmp_path / "acme.yaml").write_text(
            """
publication_name: acme
wordpress:
  site_url: https://example.com
  username: someone
  application_password: secret
api_keys:
  openai:
    api_key: publication_tier_key
""",
            encoding="utf-8",
        )
        return tmp_path

    def test_env_tier_wins_with_nothing_else_configured(self, config_dir):
        user_config = load_user_config(str(config_dir))
        pub_config = load_publication_config("acme", str(config_dir))
        # Publication config overrides openai but not mistral.
        merged = merge_configs(user_config, pub_config)
        assert merged["api_keys"]["mistral"]["api_key"] == "env_tier_mistral_key"

    def test_publication_tier_beats_env_tier(self, config_dir):
        user_config = load_user_config(str(config_dir))
        pub_config = load_publication_config("acme", str(config_dir))
        merged = merge_configs(user_config, pub_config)
        assert merged["api_keys"]["openai"]["api_key"] == "publication_tier_key"

    def test_cli_tier_beats_publication_tier(self, config_dir):
        user_config = load_user_config(str(config_dir))
        pub_config = load_publication_config("acme", str(config_dir))
        merged = merge_configs(user_config, pub_config)
        overrides = parse_api_key_overrides(["openai=cli_tier_key"])
        final = apply_api_key_overrides(merged, overrides)
        assert final["api_keys"]["openai"]["api_key"] == "cli_tier_key"
        # Untouched providers still come from whatever tier resolved them.
        assert final["api_keys"]["mistral"]["api_key"] == "env_tier_mistral_key"


class TestDotenvBeatsOsEnvThroughLoadUserConfig:
    """The actual incident, exercised through load_user_config's real code
    path rather than resolve_env in isolation: a .env-defined value must win
    over a differing OS-level variable of the same name."""

    def test_dotenv_file_value_wins_over_a_differing_os_variable(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "stale_os_value")
        monkeypatch.setattr(
            config_loader, "_EFFECTIVE_ENV", {"OPENAI_API_KEY": "dotenv_value"}
        )
        (tmp_path / "user.yaml").write_text(
            "api_keys:\n"
            "  openai:\n"
            "    api_key: ${OPENAI_API_KEY}\n"
            "  gemini:\n"
            "    api_key: dummy_gemini_key\n"
            "  mistral:\n"
            "    api_key: dummy_mistral_key\n",
            encoding="utf-8",
        )
        config = load_user_config(str(tmp_path))
        assert config["api_keys"]["openai"]["api_key"] == "dotenv_value"
