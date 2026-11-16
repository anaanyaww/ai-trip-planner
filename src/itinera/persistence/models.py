"""Relational schema for plans and their audit trail.

The checkpointer already persists the workflow's state, so this table is not a second
copy of it - it is the query surface the checkpointer does not provide. Answering
"what is the status of plan X" or "which plans are awaiting review" by deserialising
LangGraph checkpoints would mean loading an entire conversation to read one field.

The larger documents are stored as JSON rather than shredded into tables. They are
written and read whole, always by primary key, and their shape is owned by the domain
models; normalising them would buy nothing and would turn every itinerary field change
into a migration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from itinera.domain.enums import PlanStatus

_PLAN_STATUS = Enum(
    PlanStatus,
    name="plan_status",
    native_enum=False,
    length=32,
    values_callable=lambda enum: [member.value for member in enum],
)


def _utcnow() -> datetime:
    """Return the current UTC time as an aware datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every mapped class."""


class PlanRecord(Base):
    """A travel plan and its latest known artefacts.

    ``idempotency_key`` carries a unique index so a client that retries a submission -
    the usual cause being a timeout on a request that actually succeeded - gets the
    original plan back instead of starting a second, redundant workflow run.
    """

    __tablename__ = "plans"

    plan_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[PlanStatus] = mapped_column(_PLAN_STATUS, index=True)
    destination: Mapped[str] = mapped_column(String(120))
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, default=None)

    request: Mapped[dict[str, Any]] = mapped_column(JSON)
    draft: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    final: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)

    revision_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    events: Mapped[list[PlanEventRecord]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="PlanEventRecord.id",
        lazy="selectin",
    )

    __table_args__ = (Index("ix_plans_status_updated_at", "status", "updated_at"),)


class PlanEventRecord(Base):
    """One entry in a plan's audit trail.

    Kept separate from the plan row because it is append-only and because a reviewer
    asking "what happened to my plan" wants the sequence, not the latest snapshot.
    """

    __tablename__ = "plan_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("plans.plan_id", ondelete="CASCADE"), index=True
    )
    event: Mapped[str] = mapped_column(String(64))
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    plan: Mapped[PlanRecord] = relationship(back_populates="events")
