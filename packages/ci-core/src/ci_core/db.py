"""Async SQLAlchemy engine, session factory, Base, and TimestampMixin for ci-core."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ci_core.config import DatabaseSettings


def _build_url(db: DatabaseSettings) -> str:
    return f"postgresql+asyncpg://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"


def make_engine(db: DatabaseSettings) -> AsyncEngine:
    return create_async_engine(_build_url(db), pool_size=db.pool_size, echo=False)


def make_session_factory(db: DatabaseSettings) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(make_engine(db), expire_on_commit=False)


class TimestampMixin:
    """Adds created_at / updated_at columns that the database sets automatically."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a session; commit on clean exit, roll back on exception."""
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
