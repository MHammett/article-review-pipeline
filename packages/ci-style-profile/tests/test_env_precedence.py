"""bootstrap.py resolves ${VAR} placeholders in sources.yaml and user.yaml
against the same .env-beats-OS-env precedence as ci_article_review's
config_loader — they share the same configs/user.yaml and .env file, so a
stale OS-level variable must not silently override .env here either. See
ci_core.env_provenance and ci_article_review's test_api_key_precedence.py for
the same fix applied to ci-review/ci-check.
"""

from __future__ import annotations

from unittest.mock import mock_open, patch

import ci_style_profile.bootstrap as bootstrap


class TestDotenvBeatsOsEnvInUserConfig:
    def test_dotenv_value_wins_over_a_differing_os_variable(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "stale_os_value")
        monkeypatch.setattr(
            bootstrap, "_EFFECTIVE_ENV", {"OPENAI_API_KEY": "dotenv_value"}
        )
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        (configs_dir / "user.yaml").write_text(
            "api_keys:\n  openai:\n    api_key: ${OPENAI_API_KEY}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        config = bootstrap._load_user_config_lenient()
        assert config["api_keys"]["openai"]["api_key"] == "dotenv_value"


class TestDotenvBeatsOsEnvInSourcesYaml:
    """_load_sources_yaml reads a fixed path next to bootstrap.py itself, so
    the filesystem calls are mocked rather than the path — the point here is
    only that it threads _EFFECTIVE_ENV through to resolve_env_recursive."""

    def test_dotenv_value_wins_over_a_differing_os_variable(self, monkeypatch):
        monkeypatch.setenv("WP_APPLICATION_PASSWORD", "stale_os_value")
        monkeypatch.setattr(
            bootstrap, "_EFFECTIVE_ENV", {"WP_APPLICATION_PASSWORD": "dotenv_value"}
        )
        monkeypatch.setattr(
            bootstrap._yaml,
            "safe_load",
            lambda f: {
                "wordpress": {"application_password": "${WP_APPLICATION_PASSWORD}"}
            },
        )
        with (
            patch.object(bootstrap.Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
        ):
            result = bootstrap._load_sources_yaml()
        assert result["wordpress"]["application_password"] == "dotenv_value"
