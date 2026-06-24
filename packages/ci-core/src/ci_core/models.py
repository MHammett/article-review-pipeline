"""Shared ORM models — tables used across ci-core consumers."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy import JSON as _JSON
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ci_core.db import Base, TimestampMixin


class Target(Base, TimestampMixin):
    """Website or URL under active monitoring."""

    __tablename__ = "targets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    competitors: Mapped[list["Competitor"]] = relationship(back_populates="target")
    runs: Mapped[list["Run"]] = relationship(back_populates="target")
    pages: Mapped[list["Page"]] = relationship(back_populates="target")
    schedules: Mapped[list["Schedule"]] = relationship(back_populates="target")


class Competitor(Base, TimestampMixin):
    """Competitor website associated with a target."""

    __tablename__ = "competitors"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    target_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("targets.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    target: Mapped["Target"] = relationship(back_populates="competitors")
    pages: Mapped[list["Page"]] = relationship(back_populates="competitor")


class Run(Base, TimestampMixin):
    """A crawl or analysis execution."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    target_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("targets.id", ondelete="SET NULL"), nullable=True
    )
    run_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(_JSON, nullable=True)

    target: Mapped[Optional["Target"]] = relationship(back_populates="runs")
    pages: Mapped[list["Page"]] = relationship(back_populates="run")


class Page(Base, TimestampMixin):
    """A single crawled URL."""

    __tablename__ = "pages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("targets.id", ondelete="SET NULL"), nullable=True
    )
    competitor_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("competitors.id", ondelete="SET NULL"), nullable=True
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    fetched_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    run: Mapped["Run"] = relationship(back_populates="pages")
    target: Mapped[Optional["Target"]] = relationship(back_populates="pages")
    competitor: Mapped[Optional["Competitor"]] = relationship(back_populates="pages")
    extracts: Mapped[list["PageExtract"]] = relationship(back_populates="page")
    analysis_results: Mapped[list["AnalysisResult"]] = relationship(
        back_populates="page"
    )
    change_events: Mapped[list["ChangeEvent"]] = relationship(back_populates="page")


class PageExtract(Base, TimestampMixin):
    """Extracted content from a crawled page."""

    __tablename__ = "page_extracts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    page_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), nullable=False
    )
    extract_type: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(_JSON, nullable=True)

    page: Mapped["Page"] = relationship(back_populates="extracts")


class AnalysisResult(Base, TimestampMixin):
    """Output of a single analysis pass on a page."""

    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    page_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), nullable=False
    )
    analyzer: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    findings: Mapped[Optional[dict]] = mapped_column(_JSON, nullable=True)

    page: Mapped["Page"] = relationship(back_populates="analysis_results")


class DocumentRun(Base, TimestampMixin):
    """A document evaluation run (ci-article-review)."""

    __tablename__ = "document_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    source_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="article"
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    results: Mapped[list["DocumentResult"]] = relationship(back_populates="run")


class DocumentResult(Base, TimestampMixin):
    """Per-model/pass result for a document run."""

    __tablename__ = "document_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_runs.id", ondelete="CASCADE"), nullable=False
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    pass_name: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    output: Mapped[Optional[dict]] = mapped_column(_JSON, nullable=True)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    run: Mapped["DocumentRun"] = relationship(back_populates="results")


class Action(Base, TimestampMixin):
    """A triggered action or notification."""

    __tablename__ = "actions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    trigger_type: Mapped[str] = mapped_column(String(128), nullable=False)
    target_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("targets.id", ondelete="SET NULL"), nullable=True
    )
    document_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("document_runs.id", ondelete="SET NULL"), nullable=True
    )
    payload: Mapped[Optional[dict]] = mapped_column(_JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    executed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Schedule(Base, TimestampMixin):
    """Cron-style job schedule for a target."""

    __tablename__ = "schedules"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    target_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("targets.id", ondelete="SET NULL"), nullable=True
    )
    job_type: Mapped[str] = mapped_column(String(128), nullable=False)
    cron_expr: Mapped[str] = mapped_column(String(128), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    next_run_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    target: Mapped[Optional["Target"]] = relationship(back_populates="schedules")


class ChangeEvent(Base, TimestampMixin):
    """A detected content change on a page."""

    __tablename__ = "change_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    page_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), nullable=False
    )
    change_type: Mapped[str] = mapped_column(String(128), nullable=False)
    diff_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    old_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    new_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    severity: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    page: Mapped["Page"] = relationship(back_populates="change_events")
