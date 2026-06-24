"""Tests for ci_core.db — unit-level, no live database required."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ci_core.config import DatabaseSettings
from ci_core.db import (
    Base,
    TimestampMixin,
    _build_url,
    get_session,
    make_engine,
    make_session_factory,
)


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def test_build_url_format():
    db = DatabaseSettings(
        host="db.example.com",
        port=5433,
        name="mydb",
        user="appuser",
        password="s3cr3t",
    )
    url = _build_url(db)
    assert url == "postgresql+asyncpg://appuser:s3cr3t@db.example.com:5433/mydb"


def test_build_url_default_host_port():
    db = DatabaseSettings(name="ci", user="ci_user", password="ci_pass")
    url = _build_url(db)
    assert "localhost" in url
    assert ":5432/" in url


# ---------------------------------------------------------------------------
# Engine and session factory
# ---------------------------------------------------------------------------


def test_make_engine_returns_async_engine():
    db = DatabaseSettings(name="ci", user="u", password="p")
    engine = make_engine(db)
    assert isinstance(engine, AsyncEngine)


def test_make_engine_uses_pool_size():
    db = DatabaseSettings(name="ci", user="u", password="p", pool_size=10)
    engine = make_engine(db)
    assert engine.pool.size() == 10


def test_make_session_factory_returns_sessionmaker():
    db = DatabaseSettings(name="ci", user="u", password="p")
    factory = make_session_factory(db)
    assert isinstance(factory, async_sessionmaker)


# ---------------------------------------------------------------------------
# TimestampMixin and Base
# ---------------------------------------------------------------------------


def test_timestamp_mixin_columns_defined():
    assert "created_at" in TimestampMixin.__annotations__
    assert "updated_at" in TimestampMixin.__annotations__


def test_base_is_declarative():
    assert hasattr(Base, "metadata")
    assert hasattr(Base, "registry")


def test_models_registered_in_base_metadata():
    """Importing ci_core.models must register all tables in Base.metadata."""
    import ci_core.models  # noqa: F401

    expected = {
        "targets",
        "competitors",
        "runs",
        "pages",
        "page_extracts",
        "analysis_results",
        "document_runs",
        "document_results",
        "actions",
        "schedules",
        "change_events",
    }
    registered = set(Base.metadata.tables.keys())
    assert expected == registered


# ---------------------------------------------------------------------------
# get_session context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_commits_on_success():
    mock_session = AsyncMock(spec=AsyncSession)
    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    async with get_session(mock_factory) as session:
        assert session is mock_session

    mock_session.commit.assert_awaited_once()
    mock_session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_session_rolls_back_on_exception():
    mock_session = AsyncMock(spec=AsyncSession)
    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    with pytest.raises(ValueError, match="boom"):
        async with get_session(mock_factory):
            raise ValueError("boom")

    mock_session.rollback.assert_awaited_once()
    mock_session.commit.assert_not_awaited()
