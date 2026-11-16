"""Data access for plans.

The repository is the only place that knows the workflow state and the database rows
are two representations of the same plan. Nodes write state; this projects the parts
worth querying into a row after each run, so the API can answer status questions
without deserialising a checkpoint.

Projection is idempotent: running it twice with the same state produces the same row
and the same event list, which matters because a resumed workflow re-projects a plan
that already exists.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from itinera.core.exceptions import PlanNotFoundError
from itinera.core.logging import get_logger
from itinera.domain.enums import PlanStatus
from itinera.graph.state import PlanState
from itinera.persistence.models import PlanEventRecord, PlanRecord

logger = get_logger(__name__)


class PlanRepository:
    """Reads and writes plan rows."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a unit of work."""
        self._session = session

    async def get(self, plan_id: str) -> PlanRecord | None:
        """Return the plan with ``plan_id``, or ``None`` when it does not exist."""
        return await self._session.get(PlanRecord, plan_id)

    async def require(self, plan_id: str) -> PlanRecord:
        """Return the plan with ``plan_id``.

        Raises:
            PlanNotFoundError: If no such plan exists.
        """
        record = await self.get(plan_id)
        if record is None:
            raise PlanNotFoundError(f"No plan exists with id {plan_id!r}", plan_id=plan_id)
        return record

    async def find_by_idempotency_key(self, key: str) -> PlanRecord | None:
        """Return the plan a previous submission created under ``key``."""
        result = await self._session.execute(
            select(PlanRecord).where(PlanRecord.idempotency_key == key)
        )
        return result.scalar_one_or_none()

    async def list_plans(
        self, status: PlanStatus | None = None, limit: int = 50, offset: int = 0
    ) -> list[PlanRecord]:
        """Return plans newest-first, optionally filtered by status."""
        statement = select(PlanRecord).order_by(PlanRecord.updated_at.desc())
        if status is not None:
            statement = statement.where(PlanRecord.status == status)

        result = await self._session.execute(statement.limit(limit).offset(offset))
        return list(result.scalars().all())

    async def create(
        self,
        plan_id: str,
        destination: str,
        request: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> PlanRecord:
        """Insert a new plan in its initial state.

        Args:
            plan_id: Identifier shared with the checkpoint thread.
            destination: Denormalised for listing without parsing the request.
            request: The submitted travel request, serialised.
            idempotency_key: Client-supplied key de-duplicating retried submissions.

        Returns:
            The persisted record.

        Raises:
            IntegrityError: If the idempotency key is already in use. Callers should
                look the existing plan up rather than treating this as a failure.
        """
        record = PlanRecord(
            plan_id=plan_id,
            status=PlanStatus.PENDING,
            destination=destination,
            request=request,
            idempotency_key=idempotency_key,
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def project(self, state: PlanState) -> PlanRecord:
        """Write the queryable parts of a workflow state onto its row.

        Args:
            state: The state produced by a graph run.

        Returns:
            The updated record.

        Raises:
            PlanNotFoundError: If the plan row was never created.
        """
        record = await self.require(state["plan_id"])
        draft = state.get("draft")
        final = state.get("final")

        record.status = PlanStatus(state.get("status", record.status))
        record.revision_count = state.get("revision_count", record.revision_count)
        record.error = state.get("error")
        record.draft = draft if draft is not None else record.draft
        record.final = final if final is not None else record.final

        await self._sync_events(record, state.get("history", []))
        await self._session.flush()

        logger.debug(
            "repository.projected", plan_id=record.plan_id, status=record.status.value
        )
        return record

    async def _sync_events(self, record: PlanRecord, history: list[dict[str, Any]]) -> None:
        """Append history entries the row does not already carry.

        The workflow's history is append-only and cumulative, so the entries beyond the
        count already stored are exactly the new ones.
        """
        existing = len(record.events)
        for entry in history[existing:]:
            detail = {key: value for key, value in entry.items() if key not in {"event", "at"}}
            record.events.append(
                PlanEventRecord(event=str(entry.get("event", "unknown")), detail=detail)
            )

    async def mark_failed(self, plan_id: str, error: str) -> PlanRecord:
        """Record that a run failed, so the plan does not sit in a running status."""
        record = await self.require(plan_id)
        record.status = PlanStatus.FAILED
        record.error = error
        record.events.append(PlanEventRecord(event="run_failed", detail={"error": error}))
        await self._session.flush()

        logger.warning("repository.plan_failed", plan_id=plan_id, error=error)
        return record


async def create_or_reuse(
    repository: PlanRepository,
    plan_id: str,
    destination: str,
    request: dict[str, Any],
    idempotency_key: str | None,
) -> tuple[PlanRecord, bool]:
    """Create a plan, returning the existing one when the key was already used.

    Args:
        repository: Bound to the session the insert should happen in.
        plan_id: Identifier for the new plan.
        destination: Denormalised destination.
        request: The serialised travel request.
        idempotency_key: Client-supplied de-duplication key, if any.

    Returns:
        The record and whether it was newly created.
    """
    if idempotency_key:
        existing = await repository.find_by_idempotency_key(idempotency_key)
        if existing is not None:
            logger.info(
                "repository.idempotent_replay", plan_id=existing.plan_id, key=idempotency_key
            )
            return existing, False

    try:
        record = await repository.create(plan_id, destination, request, idempotency_key)
    except IntegrityError:
        if not idempotency_key:
            raise
        existing = await repository.find_by_idempotency_key(idempotency_key)
        if existing is None:  # pragma: no cover - the key must exist to have collided
            raise
        return existing, False

    return record, True
