"""The planning service: the API's view of the workflow.

This is where the two halves of the human-in-the-loop cycle meet. ``submit`` starts a
run that stops at the interrupt; ``review`` resumes that same thread from the
checkpoint with the reviewer's decision. Neither holds the other's connection open,
which is the whole point of checkpointing the pause.

Runs happen in the background. Researching a destination and drafting a week costs
tens of seconds of model latency, and a client waiting that long on a POST learns
nothing it could not learn by polling. ``POST /plan`` therefore returns an identifier
immediately and the workflow reports its stage through the plan's status.

Concurrent work on one plan is serialised with a per-plan lock. Two reviews arriving
together would otherwise resume the same checkpoint twice and race to write
contradictory state.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from itinera.core.exceptions import (
    InvalidPlanStateError,
    ItineraError,
    PlanNotFoundError,
    WorkflowError,
)
from itinera.core.logging import get_logger
from itinera.domain.enums import PlanStatus
from itinera.domain.itinerary import DraftItinerary, FinalPlan
from itinera.domain.patching import apply_patches
from itinera.domain.review import ReviewDecision
from itinera.domain.travel import TravelRequest
from itinera.graph.state import PlanState, initial_state
from itinera.persistence.models import PlanRecord
from itinera.persistence.repository import PlanRepository, create_or_reuse
from itinera.services.progress import running
from itinera.services.runtime import Runtime

logger = get_logger(__name__)


@asynccontextmanager
async def _reporting(plan_id: str) -> AsyncIterator[None]:
    """Bind live progress reporting to ``plan_id`` for the duration of a run."""
    with running(plan_id):
        yield


_INTERRUPT_KEY = "__interrupt__"


class PlanningService:
    """Starts, inspects and resumes planning workflows."""

    def __init__(self, runtime: Runtime) -> None:
        """Bind the service to a started runtime."""
        self._runtime = runtime
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, plan_id: str) -> asyncio.Lock:
        """Return the lock serialising work on ``plan_id``."""
        return self._locks.setdefault(plan_id, asyncio.Lock())

    @staticmethod
    def _config(plan_id: str) -> RunnableConfig:
        """Return the LangGraph config addressing a plan's checkpoint thread.

        The thread id is the plan id, so a plan's checkpoints and its API resource are
        the same thing addressed two ways.
        """
        return {"configurable": {"thread_id": plan_id}}


    async def submit(
        self, request: TravelRequest, idempotency_key: str | None = None
    ) -> tuple[str, bool]:
        """Register a new plan and return its identifier.

        The request is checked against the configured ceilings here rather than only
        inside the workflow, so an over-long trip is rejected by the API with a 422
        instead of being accepted and failing in the background.

        Args:
            request: The validated travel request.
            idempotency_key: Client key de-duplicating a retried submission.

        Returns:
            The plan identifier and whether this call created it.

        Raises:
            PlanValidationError: If the request exceeds a configured ceiling.
        """
        request.validate_against_limits(self._runtime.settings)

        async with self._runtime.database.session() as session:
            record, created = await create_or_reuse(
                PlanRepository(session),
                plan_id=str(uuid.uuid4()),
                destination=request.destination,
                request=request.model_dump(mode="json"),
                idempotency_key=idempotency_key,
            )
            plan_id = record.plan_id

        logger.info("planning.submitted", plan_id=plan_id, created=created, **request.summary())
        return plan_id, created

    async def run(self, plan_id: str, request: TravelRequest) -> None:
        """Execute the workflow up to the review pause.

        Intended to be scheduled in the background. Failures are recorded on the plan
        rather than raised, because there is no caller left to receive them.
        """
        async with self._lock_for(plan_id), _reporting(plan_id):
            try:
                await self._runtime.graph.ainvoke(
                    initial_state(plan_id, request), config=self._config(plan_id)
                )
            except GraphInterrupt:
                logger.info("planning.paused_for_review", plan_id=plan_id)
            except ItineraError as exc:
                await self._record_failure(plan_id, exc.detail)
                return
            except Exception as exc:
                logger.exception("planning.run_failed", plan_id=plan_id)
                await self._record_failure(plan_id, f"The workflow failed: {exc}")
                return

            await self._project(plan_id)

    async def review(self, plan_id: str, decision: ReviewDecision) -> PlanRecord:
        """Resume a paused workflow with the reviewer's decision.

        Args:
            plan_id: The plan under review.
            decision: Approve, reject with feedback, or modify.

        Returns:
            The plan record as it stands after the decision was applied.

        Raises:
            PlanNotFoundError: If the plan does not exist.
            InvalidPlanStateError: If the workflow is not paused for a review.
            WorkflowError: If resuming the workflow failed.
        """
        async with self._lock_for(plan_id), _reporting(plan_id):
            record = await self._require_record(plan_id)
            if not record.status.accepts_review:
                raise InvalidPlanStateError(
                    f"Plan {plan_id} is {record.status.value} and is not awaiting review.",
                    current_status=record.status.value,
                )

            self._precheck_patches(record, decision)

            try:
                await self._runtime.graph.ainvoke(
                    Command(resume=decision.model_dump(mode="json")),
                    config=self._config(plan_id),
                )
            except GraphInterrupt:
                logger.info("planning.paused_for_review", plan_id=plan_id)
            except ItineraError:
                await self._resync(plan_id)
                raise
            except Exception as exc:
                logger.exception("planning.review_failed", plan_id=plan_id)
                await self._resync(plan_id)
                raise WorkflowError(
                    f"Resuming plan {plan_id} failed: {exc}. The draft is unchanged; "
                    "submit the review again.",
                    plan_id=plan_id,
                ) from exc

            return await self._project(plan_id)

    @staticmethod
    def _precheck_patches(record: PlanRecord, decision: ReviewDecision) -> None:
        """Reject unapplyable patches before the workflow is resumed.

        A patch naming a day that does not exist is a client mistake, and the client
        should hear about it as a 422 rather than a 200 carrying an error note. Doing
        it here also leaves the interrupt unconsumed, so the plan stays exactly where
        it was and the reviewer can simply correct the patch and resubmit.

        Raises:
            PlanValidationError: If any patch addresses something that does not exist.
        """
        if not decision.is_structured_modification or record.draft is None:
            return

        apply_patches(DraftItinerary.model_validate(record.draft), decision.patches)


    async def get(self, plan_id: str) -> PlanRecord:
        """Return a plan.

        Raises:
            PlanNotFoundError: If the plan does not exist.
        """
        return await self._require_record(plan_id)

    async def get_final(self, plan_id: str) -> FinalPlan:
        """Return the finalised plan.

        Raises:
            PlanNotFoundError: If the plan does not exist.
            InvalidPlanStateError: If the plan has not been approved yet.
        """
        record = await self._require_record(plan_id)
        if record.final is None:
            raise InvalidPlanStateError(
                f"Plan {plan_id} has no final plan yet; it is {record.status.value}. "
                "The final plan is available once the draft has been approved.",
                current_status=record.status.value,
            )
        return FinalPlan.model_validate(record.final)

    async def list_plans(
        self, status: PlanStatus | None, limit: int, offset: int
    ) -> list[PlanRecord]:
        """Return plans newest-first, optionally filtered by status."""
        async with self._runtime.database.session() as session:
            return await PlanRepository(session).list_plans(status, limit, offset)

    async def pending_review(self, plan_id: str) -> dict[str, Any] | None:
        """Return the payload the workflow surfaced for review, if it is paused.

        Read from the checkpoint rather than from the plan row, because this is the
        interrupt's own payload - what the workflow actually asked a human to look at.
        """
        snapshot = await self._runtime.graph.aget_state(self._config(plan_id))
        for task in snapshot.tasks:
            for pause in task.interrupts:
                value = pause.value
                if isinstance(value, dict):
                    return value
        return None


    async def _state(self, plan_id: str) -> PlanState:
        """Return the checkpointed workflow state for ``plan_id``."""
        snapshot = await self._runtime.graph.aget_state(self._config(plan_id))
        values: Any = snapshot.values
        return cast(PlanState, values) if isinstance(values, dict) else PlanState()

    async def _project(self, plan_id: str) -> PlanRecord:
        """Write the current workflow state onto the plan row."""
        state = await self._state(plan_id)
        state["plan_id"] = plan_id

        async with self._runtime.database.session() as session:
            return await PlanRepository(session).project(state)

    async def _resync(self, plan_id: str) -> None:
        """Re-project the plan from its checkpoint after a failed review.

        A node that raises commits no writes, so the checkpoint still holds the pause
        and the draft the reviewer was looking at. Marking the plan ``failed`` would
        throw away a perfectly good draft over a transient provider error - a rate
        limit during a revision is a reason to retry, not to lose the trip. The row is
        instead brought back into line with the checkpoint, which still says the plan
        is awaiting review.
        """
        try:
            await self._project(plan_id)
        except Exception:  # pragma: no cover - the row exists; this is belt and braces
            logger.exception("planning.resync_failed", plan_id=plan_id)

    async def _record_failure(self, plan_id: str, detail: str) -> None:
        """Mark a plan failed, so it does not sit forever in a running status."""
        async with self._runtime.database.session() as session:
            await PlanRepository(session).mark_failed(plan_id, detail)

    async def _require_record(self, plan_id: str) -> PlanRecord:
        """Return the plan row, or explain that it does not exist."""
        async with self._runtime.database.session() as session:
            record = await PlanRepository(session).get(plan_id)
        if record is None:
            raise PlanNotFoundError(f"No plan exists with id {plan_id!r}", plan_id=plan_id)
        return record
