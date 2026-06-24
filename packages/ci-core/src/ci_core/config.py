"""Application settings using pydantic-settings.

Load order: defaults → .env file → environment variables.
Nested settings use double-underscore delimiter: DATABASE__HOST, LLM__ANTHROPIC__API_KEY.
Missing required fields raise ValidationError at Settings() construction time.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str
    user: str
    password: str
    pool_size: int = 5


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379"
    max_connections: int = 10


class ProviderConfig(BaseModel):
    api_key: str = ""
    model: str = ""
    timeout: int = 30
    max_retries: int = 3


class LLMSettings(BaseModel):
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    mistral: ProviderConfig = Field(default_factory=ProviderConfig)
    grok: ProviderConfig = Field(default_factory=ProviderConfig)


class AppSettings(BaseModel):
    env: str = "development"
    debug: bool = False
    allowed_origins: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Top-level settings — composes all subsystem configs.

    Required env vars (no default): DATABASE__NAME, DATABASE__USER, DATABASE__PASSWORD.
    All others have sensible defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    database: DatabaseSettings
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    app: AppSettings = Field(default_factory=AppSettings)


@lru_cache
def get_settings() -> Settings:
    """Return the singleton Settings instance, constructed once and cached."""
    return Settings()  # type: ignore[call-arg]  # pydantic-settings reads fields from env vars
