"""Tests for ci_core.config — all external state controlled via monkeypatch."""

import pytest

# ci_core.config/.db/.models/.logging moved behind the `persistence` extra
# (audit finding 13): they have no production consumer, and making them
# mandatory meant every CLI user installed an async PostgreSQL driver and an
# ASGI web framework for a tool that never serves HTTP. Skip cleanly when the
# extra is absent; CI installs it, so coverage is unchanged there.
pytest.importorskip(
    "pydantic_settings",
    reason="ci-core[persistence] not installed — uv sync --extra persistence",
)


import pytest
from pydantic import ValidationError

from ci_core.config import (
    AppSettings,
    LLMSettings,
    ProviderConfig,
    RedisSettings,
    Settings,
    get_settings,
)

# Minimal env required by Settings (DATABASE__NAME/USER/PASSWORD have no default).
_DB_ENV = {
    "DATABASE__NAME": "testdb",
    "DATABASE__USER": "testuser",
    "DATABASE__PASSWORD": "testpass",
}


# ---------------------------------------------------------------------------
# Individual model defaults
# ---------------------------------------------------------------------------


def test_redis_defaults():
    s = RedisSettings()
    assert s.url == "redis://localhost:6379"
    assert s.max_connections == 10


def test_llm_has_all_providers():
    s = LLMSettings()
    for provider in ("anthropic", "openai", "gemini", "mistral", "grok"):
        p = getattr(s, provider)
        assert isinstance(p, ProviderConfig)
        assert p.timeout == 30
        assert p.max_retries == 3
        assert p.api_key == ""
        assert p.model == ""


def test_app_defaults():
    s = AppSettings()
    assert s.env == "development"
    assert s.debug is False
    assert s.allowed_origins == []


# ---------------------------------------------------------------------------
# Settings construction with required database fields
# ---------------------------------------------------------------------------


def test_settings_requires_database_fields(monkeypatch):
    """Missing DATABASE__NAME/USER/PASSWORD must raise ValidationError."""
    monkeypatch.delenv("DATABASE__NAME", raising=False)
    monkeypatch.delenv("DATABASE__USER", raising=False)
    monkeypatch.delenv("DATABASE__PASSWORD", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_settings_database_env_override(monkeypatch):
    for k, v in _DB_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DATABASE__HOST", "db.prod.internal")
    monkeypatch.setenv("DATABASE__PORT", "5433")
    monkeypatch.setenv("DATABASE__POOL_SIZE", "20")
    s = Settings()
    assert s.database.host == "db.prod.internal"
    assert s.database.port == 5433
    assert s.database.name == "testdb"
    assert s.database.user == "testuser"
    assert s.database.password == "testpass"
    assert s.database.pool_size == 20


def test_settings_redis_env_override(monkeypatch):
    for k, v in _DB_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("REDIS__URL", "redis://cache.internal:6380")
    monkeypatch.setenv("REDIS__MAX_CONNECTIONS", "50")
    s = Settings()
    assert s.redis.url == "redis://cache.internal:6380"
    assert s.redis.max_connections == 50


def test_settings_llm_env_override(monkeypatch):
    for k, v in _DB_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("LLM__ANTHROPIC__API_KEY", "sk-ant-test")
    monkeypatch.setenv("LLM__ANTHROPIC__MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("LLM__OPENAI__API_KEY", "sk-oai-test")
    monkeypatch.setenv("LLM__OPENAI__TIMEOUT", "60")
    s = Settings()
    assert s.llm.anthropic.api_key == "sk-ant-test"
    assert s.llm.anthropic.model == "claude-sonnet-4-6"
    assert s.llm.openai.api_key == "sk-oai-test"
    assert s.llm.openai.timeout == 60
    # Untouched providers retain defaults
    assert s.llm.gemini.api_key == ""
    assert s.llm.mistral.max_retries == 3


def test_settings_app_env_override(monkeypatch):
    for k, v in _DB_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("APP__ENV", "production")
    monkeypatch.setenv("APP__DEBUG", "false")
    monkeypatch.setenv(
        "APP__ALLOWED_ORIGINS", '["https://example.com","https://api.example.com"]'
    )
    s = Settings()
    assert s.app.env == "production"
    assert s.app.debug is False
    assert s.app.allowed_origins == ["https://example.com", "https://api.example.com"]


def test_settings_debug_true(monkeypatch):
    for k, v in _DB_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("APP__DEBUG", "true")
    s = Settings()
    assert s.app.debug is True


# ---------------------------------------------------------------------------
# get_settings() singleton / lru_cache behaviour
# ---------------------------------------------------------------------------


def test_get_settings_returns_singleton(monkeypatch):
    for k, v in _DB_ENV.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()


def test_get_settings_cache_survives_env_change(monkeypatch):
    """Once cached, env changes do not affect the returned instance."""
    for k, v in _DB_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("APP__ENV", "staging")
    get_settings.cache_clear()
    s1 = get_settings()
    monkeypatch.setenv("APP__ENV", "production")
    s2 = get_settings()
    assert s1 is s2
    assert s1.app.env == "staging"
    get_settings.cache_clear()
