"""Tests for the planning service's handling of workflow outcomes.

The service is what turns a graph run into a plan's recorded state, so it has to tell
three outcomes apart: the workflow finished, the workflow paused for review, and the
workflow failed. Conflating the middle one with the last marks a perfectly good paused
plan as `failed`, which is what happened on the first live run - LangGraph usually
absorbs its own interrupt signal, but it reaches the caller under some execution
contexts, and offline runs never surfaced it.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from itinera.core.exceptions import PlanNotFoundError, WorkflowError
from itinera.domain.enums import PlanStatus
from itinera.domain.travel import TravelRequest
from itinera.services.planning import PlanningService
from itinera.services.runtime import Runtime


def _raise_interrupt(*_args: Any, **_kwargs: Any) -> Any:
    """Stand in for a graph run that pauses instead of returning."""

    async def _invoke(*_a: Any, **_kw: Any) -> Any:
        raise GraphInterrupt((Interrupt(value={"paused": True}, id="i1"),))

    return _invoke()


async def test_a_pausing_run_is_not_recorded_as_a_failure(
    started_runtime: Runtime, travel_request: TravelRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GraphInterrupt reaching the service means paused, not failed."""
    service = PlanningService(started_runtime)
    plan_id, _ = await service.submit(travel_request)

    await service.run(plan_id, travel_request)
    monkeypatch.setattr(started_runtime.graph, "ainvoke", _raise_interrupt)
    await service.run(plan_id, travel_request)

    record = await service.get(plan_id)
    assert record.status is PlanStatus.AWAITING_REVIEW
    assert record.error is None
    assert record.draft is not None


async def test_a_pausing_review_is_not_recorded_as_a_failure(
    started_runtime: Runtime, travel_request: TravelRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from itinera.domain.review import ReviewDecision

    service = PlanningService(started_runtime)
    plan_id, _ = await service.submit(travel_request)
    await service.run(plan_id, travel_request)

    monkeypatch.setattr(started_runtime.graph, "ainvoke", _raise_interrupt)
    record = await service.review(
        plan_id, ReviewDecision(action="reject", feedback="Day 2 is too packed.")
    )

    assert record.status is PlanStatus.AWAITING_REVIEW
    assert record.error is None


async def test_a_genuine_failure_is_still_recorded(
    started_runtime: Runtime, travel_request: TravelRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pause handling must not swallow real errors."""

    def _explode(*_a: Any, **_kw: Any) -> Any:
        async def _invoke(*_aa: Any, **_kk: Any) -> Any:
            raise RuntimeError("the provider is down")

        return _invoke()

    service = PlanningService(started_runtime)
    plan_id, _ = await service.submit(travel_request)

    monkeypatch.setattr(started_runtime.graph, "ainvoke", _explode)
    await service.run(plan_id, travel_request)

    record = await service.get(plan_id)
    assert record.status is PlanStatus.FAILED
    assert "the provider is down" in (record.error or "")


async def test_a_failing_review_raises_without_failing_the_plan(
    started_runtime: Runtime, travel_request: TravelRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller is told the review failed; the plan itself survives to be retried."""
    from itinera.domain.review import ReviewDecision

    def _explode(*_a: Any, **_kw: Any) -> Any:
        async def _invoke(*_aa: Any, **_kk: Any) -> Any:
            raise RuntimeError("resume exploded")

        return _invoke()

    service = PlanningService(started_runtime)
    plan_id, _ = await service.submit(travel_request)
    await service.run(plan_id, travel_request)

    monkeypatch.setattr(started_runtime.graph, "ainvoke", _explode)
    with pytest.raises(WorkflowError):
        await service.review(plan_id, ReviewDecision(action="approve"))

    record = await service.get(plan_id)
    assert record.status is PlanStatus.AWAITING_REVIEW
    assert record.draft is not None


async def test_an_unknown_plan_is_reported(started_runtime: Runtime) -> None:
    service = PlanningService(started_runtime)

    with pytest.raises(PlanNotFoundError):
        await service.get("does-not-exist")


async def test_a_failed_review_keeps_the_draft_reviewable(
    started_runtime: Runtime, travel_request: TravelRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider hiccup during a revision must not cost the traveller their draft.

    A node that raises commits no writes, so the checkpoint still holds the pause. The
    plan therefore stays awaiting_review and the reviewer can simply submit again -
    rather than being told the whole trip failed because one call was rate limited.
    """
    from itinera.domain.review import ReviewDecision

    def _rate_limited(*_a: Any, **_kw: Any) -> Any:
        async def _invoke(*_aa: Any, **_kk: Any) -> Any:
            raise RuntimeError("Error code: 413 - Request too large")

        return _invoke()

    service = PlanningService(started_runtime)
    plan_id, _ = await service.submit(travel_request)
    await service.run(plan_id, travel_request)
    before = await service.get(plan_id)

    monkeypatch.setattr(started_runtime.graph, "ainvoke", _rate_limited)
    with pytest.raises(WorkflowError, match="submit the review again"):
        await service.review(
            plan_id, ReviewDecision(action="reject", feedback="Day 2 is too packed.")
        )

    after = await service.get(plan_id)
    assert after.status is PlanStatus.AWAITING_REVIEW
    assert after.draft == before.draft
