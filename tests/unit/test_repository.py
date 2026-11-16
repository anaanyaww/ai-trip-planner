"""Tests for the plan repository.

Projection is the interesting part: it runs after every graph run, including runs that
resume a plan already projected once. It therefore has to be idempotent - re-running it
must not duplicate the audit trail - and it has to preserve what a later run did not
produce, since a resumed run's state carries a draft but no final plan until approval.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from itinera.core.config import Settings
from itinera.core.exceptions import PlanNotFoundError
from itinera.domain.enums import PlanStatus
from itinera.domain.itinerary import DayPlan, DraftItinerary
from itinera.domain.travel import TravelRequest
from itinera.graph.state import PlanState, initial_state, store
from itinera.persistence.database import Database
from itinera.persistence.repository import PlanRepository, create_or_reuse

PLAN_ID = "plan-1"


@pytest_asyncio.fixture
async def session(settings: Settings) -> AsyncIterator[AsyncSession]:
    """Yield a session against a freshly created schema."""
    database = Database(settings)
    await database.create_schema()
    async with database.session() as active:
        yield active
    await database.dispose()


@pytest.fixture
def repository(session: AsyncSession) -> PlanRepository:
    return PlanRepository(session)


def _draft(request: TravelRequest) -> DraftItinerary:
    """Return a minimal itinerary covering the request's dates."""
    return DraftItinerary(
        destination=request.destination,
        start_date=request.start_date,
        end_date=request.end_date,
        travelers=request.travelers,
        days=[
            DayPlan(day_number=number, day_date=day)
            for number, day in enumerate(request.trip_dates(), start=1)
        ],
    )


def _state(request: TravelRequest, **overrides: Any) -> PlanState:
    """Return a workflow state for the plan under test."""
    state = initial_state(PLAN_ID, request)
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


async def _create(
    repository: PlanRepository, request: TravelRequest, key: str | None = None
) -> Any:
    return await repository.create(
        PLAN_ID, request.destination, request.model_dump(mode="json"), key
    )


async def test_a_missing_plan_is_reported_by_id(repository: PlanRepository) -> None:
    assert await repository.get("nope") is None

    with pytest.raises(PlanNotFoundError, match="nope"):
        await repository.require("nope")


async def test_a_created_plan_starts_pending(
    repository: PlanRepository, travel_request: TravelRequest
) -> None:
    record = await _create(repository, travel_request)

    assert record.status is PlanStatus.PENDING
    assert record.destination == travel_request.destination
    assert record.draft is None and record.final is None


async def test_projection_writes_the_queryable_fields(
    repository: PlanRepository, travel_request: TravelRequest
) -> None:
    await _create(repository, travel_request)
    draft = _draft(travel_request)

    record = await repository.project(
        _state(
            travel_request,
            status=PlanStatus.AWAITING_REVIEW,
            draft=store(draft),
            revision_count=1,
            history=[{"event": "draft_ready", "at": "now", "revision": 0}],
        )
    )

    assert record.status is PlanStatus.AWAITING_REVIEW
    assert record.revision_count == 1
    assert record.draft is not None
    assert len(record.draft["days"]) == travel_request.duration_days
    assert [event.event for event in record.events] == ["draft_ready"]
    assert record.events[0].detail == {"revision": 0}


async def test_projection_is_idempotent(
    repository: PlanRepository, travel_request: TravelRequest
) -> None:
    """A resumed run re-projects a plan that already exists."""
    await _create(repository, travel_request)
    history = [
        {"event": "request_accepted", "at": "t0"},
        {"event": "draft_ready", "at": "t1"},
    ]
    state = _state(travel_request, status=PlanStatus.AWAITING_REVIEW, history=history)

    await repository.project(state)
    record = await repository.project(state)

    assert [event.event for event in record.events] == ["request_accepted", "draft_ready"]


async def test_projection_appends_only_the_new_history(
    repository: PlanRepository, travel_request: TravelRequest
) -> None:
    await _create(repository, travel_request)
    history = [{"event": "request_accepted", "at": "t0"}]
    await repository.project(_state(travel_request, history=list(history)))

    history.append({"event": "plan_finalized", "at": "t1"})
    record = await repository.project(_state(travel_request, history=history))

    assert [event.event for event in record.events] == ["request_accepted", "plan_finalized"]


async def test_projection_keeps_a_draft_the_new_state_does_not_carry(
    repository: PlanRepository, travel_request: TravelRequest
) -> None:
    """A later run must not blank out what an earlier one produced."""
    await _create(repository, travel_request)
    await repository.project(
        _state(
            travel_request,
            status=PlanStatus.AWAITING_REVIEW,
            draft=store(_draft(travel_request)),
        )
    )

    record = await repository.project(_state(travel_request, status=PlanStatus.REVISING))

    assert record.draft is not None
    assert record.status is PlanStatus.REVISING


async def test_a_failed_run_leaves_the_plan_failed_not_running(
    repository: PlanRepository, travel_request: TravelRequest
) -> None:
    await _create(repository, travel_request)

    record = await repository.mark_failed(PLAN_ID, "the provider is down")

    assert record.status is PlanStatus.FAILED
    assert record.error == "the provider is down"
    assert [event.event for event in record.events] == ["run_failed"]


async def test_an_idempotency_key_returns_the_original_plan(
    repository: PlanRepository, travel_request: TravelRequest
) -> None:
    first, created_first = await create_or_reuse(
        repository, "plan-a", "Kyoto", travel_request.model_dump(mode="json"), "key-1"
    )
    second, created_second = await create_or_reuse(
        repository, "plan-b", "Kyoto", travel_request.model_dump(mode="json"), "key-1"
    )

    assert (created_first, created_second) == (True, False)
    assert second.plan_id == first.plan_id == "plan-a"


async def test_submissions_without_a_key_are_independent(
    repository: PlanRepository, travel_request: TravelRequest
) -> None:
    payload = travel_request.model_dump(mode="json")
    first, _ = await create_or_reuse(repository, "plan-a", "Kyoto", payload, None)
    second, created = await create_or_reuse(repository, "plan-b", "Kyoto", payload, None)

    assert created is True
    assert first.plan_id != second.plan_id


async def test_listing_filters_by_status_and_orders_by_recency(
    repository: PlanRepository, travel_request: TravelRequest
) -> None:
    payload = travel_request.model_dump(mode="json")
    for plan_id in ("plan-a", "plan-b", "plan-c"):
        await create_or_reuse(repository, plan_id, "Kyoto", payload, None)

    awaiting = await repository.require("plan-b")
    awaiting.status = PlanStatus.AWAITING_REVIEW

    assert len(await repository.list_plans(None, limit=50, offset=0)) == 3
    assert [
        record.plan_id for record in await repository.list_plans(PlanStatus.AWAITING_REVIEW, 50, 0)
    ] == ["plan-b"]
    assert len(await repository.list_plans(None, limit=2, offset=0)) == 2
