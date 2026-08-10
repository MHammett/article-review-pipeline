"""Alembic env.py — async SQLAlchemy + ci_core models."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from ci_core.config import get_settings
from ci_core.db import Base

# Register all models so autogenerate can see them.
import ci_core.models  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _build_url() -> str:
    db = get_settings().database
    return f"postgresql+asyncpg://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"


def run_migrations_offline() -> None:
    """Generate SQL without a live database connection."""
    context.configure(
        url=_build_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live database using an async engine."""
    engine = create_async_engine(_build_url(), echo=False)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
