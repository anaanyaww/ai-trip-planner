"""Tests for the orchestrator, driven through the compiled graph.

These exercise the assignment's core claim: that the workflow genuinely pauses, that
its state survives the pause, and that each of the three review actions resumes it
into the right branch. The graph is driven directly rather than over HTTP so a failure
points at the workflow rather than at the transport.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.types import Command

from itinera.agents.schemas import PlannedActivity, PlannerOutput, RoutingDecision
from itinera.domain.enums import PlanStatus, RoutingTarget
from itinera.domain.itinerary import DraftItinerary, FinalPlan
from itinera.domain.travel import TravelRequest
from itinera.graph.state import initial_state, load
from itinera.llm.providers.fake import FakeChatModel
from itinera.services.runtime import Runtime

PLAN_ID = "test-plan"


def _config(plan_id: str = PLAN_ID) -> dict[str, Any]:
    return {"configurable": {"thread_id": plan_id}}


def _planned(*names: str) -> PlannerOutput:
    """Return a planner output with one day per named activity."""
    return PlannerOutput(
        summary="A short trip.",
        day_themes=list(names),
        activities=[
            PlannedActivity(day=position, name=name, cost_per_person=20.0)
            for position, name in enumerate(names, start=1)
        ],
        highlights=list(names),
    )


async def _start(runtime: Runtime, request: TravelRequest) -> dict[str, Any]:
    """Run the workflow up to its first pause."""
    return await runtime.graph.ainvoke(initial_state(PLAN_ID, request), config=_config())


async def _state(runtime: Runtime) -> dict[str, Any]:
    """Return the checkpointed state."""
    snapshot = await runtime.graph.aget_state(_config())
    return dict(snapshot.values)


def _draft(state: dict[str, Any]) -> DraftItinerary:
    """Rebuild the draft the state stores as JSON."""
    draft = load(DraftItinerary, state.get("draft"))
    assert draft is not None
    return draft


def _final(state: dict[str, Any]) -> FinalPlan:
    """Rebuild the final plan the state stores as JSON."""
    final = load(FinalPlan, state.get("final"))
    assert final is not None
    return final


async def _resume(runtime: Runtime, **decision: Any) -> dict[str, Any]:
    """Resume the paused workflow with a review decision."""
    return await runtime.graph.ainvoke(Command(resume=decision), config=_config())


async def test_workflow_pauses_for_review(
    started_runtime: Runtime, travel_request: TravelRequest
) -> None:
    await _start(started_runtime, travel_request)

    snapshot = await started_runtime.graph.aget_state(_config())
    assert snapshot.next == ("await_review",), "the graph must stop at the review node"
    assert snapshot.values["status"] is PlanStatus.AWAITING_REVIEW
    assert snapshot.values["draft"] is not None


async def test_pause_surfaces_the_draft_for_review(
    started_runtime: Runtime, travel_request: TravelRequest, fake_model: FakeChatModel
) -> None:
    """The interrupt's payload is what a human is asked to look at."""
    fake_model.structured_responses = [_planned("Nijo Castle", "Nishiki Market", "Arashiyama")]

    await _start(started_runtime, travel_request)
    snapshot = await started_runtime.graph.aget_state(_config())
    payload = snapshot.tasks[0].interrupts[0].value

    assert payload["plan_id"] == PLAN_ID
    assert payload["revisions_remaining"] == started_runtime.settings.max_revisions
    assert "Nijo Castle" in payload["rendered"]
    assert len(payload["itinerary"]["days"]) == travel_request.duration_days


async def test_state_survives_the_pause(
    started_runtime: Runtime, travel_request: TravelRequest
) -> None:
    """The checkpoint holds everything needed to resume, not just the draft."""
    await _start(started_runtime, travel_request)
    state = await _state(started_runtime)

    assert state["request"]["destination"] == travel_request.destination
    assert state["brief"] is not None
    assert state["plan_id"] == PLAN_ID
    assert [entry["event"] for entry in state["history"]] == [
        "request_accepted",
        "research_complete",
        "draft_ready",
    ]


async def test_days_are_numbered_and_dated_from_the_request(
    started_runtime: Runtime, travel_request: TravelRequest, fake_model: FakeChatModel
) -> None:
    """Numbering is assigned by the node, so the model cannot get it wrong."""
    fake_model.structured_responses = [_planned("one", "two", "three", "four", "five")]

    await _start(started_runtime, travel_request)
    draft = _draft(await _state(started_runtime))

    assert [day.day_number for day in draft.days] == list(
        range(1, travel_request.duration_days + 1)
    )
    assert draft.days[0].day_date == travel_request.start_date
    assert draft.days[-1].day_date == travel_request.end_date


async def test_approval_produces_the_final_plan(
    started_runtime: Runtime, travel_request: TravelRequest
) -> None:
    await _start(started_runtime, travel_request)
    await _resume(started_runtime, action="approve")

    state = await _state(started_runtime)
    assert state["status"] is PlanStatus.COMPLETED
    assert _final(state).revision_limit_reached is False

    snapshot = await started_runtime.graph.aget_state(_config())
    assert snapshot.next == (), "an approved plan is finished"


async def test_rejection_routes_back_and_pauses_again(
    started_runtime: Runtime, travel_request: TravelRequest
) -> None:
    await _start(started_runtime, travel_request)
    await _resume(started_runtime, action="reject", feedback="Day 2 is far too packed.")

    snapshot = await started_runtime.graph.aget_state(_config())
    state = dict(snapshot.values)

    assert snapshot.next == ("await_review",), "a revised draft is presented again"
    assert state["revision_count"] == 1
    assert _draft(state).revision == 1
    assert state["feedback"] is None, "feedback is consumed by the revision"


async def test_rejection_needing_facts_is_routed_to_research(
    started_runtime: Runtime, travel_request: TravelRequest, fake_model: FakeChatModel
) -> None:
    """A complaint about missing facts re-runs research, not just planning."""
    fake_model.structured_responses = [
        RoutingDecision(target="research", reason="The brief covers no food markets.")
    ]

    await _start(started_runtime, travel_request)
    await _resume(started_runtime, action="reject", feedback="Nothing covers the food markets.")

    state = await _state(started_runtime)
    assert state["routing"] == RoutingTarget.RESEARCH.value
    assert [entry["event"] for entry in state["history"]].count("research_complete") == 2


async def test_rejection_about_arrangement_stays_with_the_planner(
    started_runtime: Runtime, travel_request: TravelRequest, fake_model: FakeChatModel
) -> None:
    fake_model.structured_responses = [
        RoutingDecision(target="planner", reason="Only the pacing is wrong.")
    ]

    await _start(started_runtime, travel_request)
    await _resume(started_runtime, action="reject", feedback="Day 2 is exhausting.")

    state = await _state(started_runtime)
    assert state["routing"] == RoutingTarget.PLANNER.value
    assert [entry["event"] for entry in state["history"]].count("research_complete") == 1


async def test_unroutable_feedback_falls_back_to_the_planner(
    started_runtime: Runtime, travel_request: TravelRequest, fake_model: FakeChatModel
) -> None:
    """Replanning is the cheaper path, so an unusable classification takes it."""
    fake_model.structured_responses = [RoutingDecision(target="nonsense")]

    await _start(started_runtime, travel_request)
    await _resume(started_runtime, action="reject", feedback="Something is off.")

    assert (await _state(started_runtime))["routing"] == RoutingTarget.PLANNER.value


async def test_structured_modification_is_applied_without_replanning(
    started_runtime: Runtime, travel_request: TravelRequest, fake_model: FakeChatModel
) -> None:
    fake_model.structured_responses = [_planned("one", "two", "three", "four", "five")]
    await _start(started_runtime, travel_request)

    await _resume(
        started_runtime,
        action="modify",
        patches=[
            {
                "operation": "set_day_notes",
                "day_number": 2,
                "value": "Museum day; book ahead.",
            }
        ],
    )

    draft = _draft(await _state(started_runtime))
    state = await _state(started_runtime)
    assert draft.day(2).notes == "Museum day; book ahead."
    assert draft.day(1).theme == "one", "untouched days must be untouched"
    assert [entry["event"] for entry in state["history"]].count("draft_ready") == 1


async def test_modification_by_instruction_replans(
    started_runtime: Runtime, travel_request: TravelRequest
) -> None:
    await _start(started_runtime, travel_request)
    await _resume(
        started_runtime,
        action="modify",
        instruction="Swap the day 4 museum for something outdoors.",
    )

    state = await _state(started_runtime)
    assert [entry["event"] for entry in state["history"]].count("draft_ready") == 2
    assert _draft(state).revision == 1
    assert state["instruction"] is None, "the instruction is consumed by the revision"


async def test_a_modified_draft_is_shown_before_it_can_be_finalised(
    started_runtime: Runtime, travel_request: TravelRequest
) -> None:
    """Nothing reaches the final plan without a human having seen it."""
    await _start(started_runtime, travel_request)
    await _resume(
        started_runtime,
        action="modify",
        patches=[{"operation": "set_lodging_preference", "value": "Near Gion"}],
    )

    snapshot = await started_runtime.graph.aget_state(_config())
    assert snapshot.next == ("await_review",)
    assert snapshot.values["final"] is None


@pytest.mark.parametrize("limit", [0, 2])
async def test_the_revision_ceiling_ends_the_loop(
    runtime_factory: Any, travel_request: TravelRequest, limit: int
) -> None:
    """A reviewer who keeps rejecting ends up with a plan rather than a loop."""
    runtime = runtime_factory(max_revisions=limit)
    await runtime.start()
    try:
        await _start(runtime, travel_request)
        for _ in range(limit + 1):
            await runtime.graph.ainvoke(
                Command(resume={"action": "reject", "feedback": "Still not right."}),
                config=_config(),
            )

        snapshot = await runtime.graph.aget_state(_config())
        state = dict(snapshot.values)
        assert snapshot.next == ()
        assert state["status"] is PlanStatus.COMPLETED
        assert _final(state).revision_limit_reached is True
        assert state["revision_limit_reached"] is True
    finally:
        await runtime.stop()


async def test_approval_carries_the_preparation_material_into_the_final_plan(
    started_runtime: Runtime, travel_request: TravelRequest, fake_model: FakeChatModel
) -> None:
    """The finaliser's output must reach the plan, not just be generated and dropped.

    Offline the model returns an empty stub, so without scripting a response this path
    looks identical whether the wiring works or not.
    """
    from itinera.agents.schemas import TripPreparation

    fake_model.structured_responses = [
        TripPreparation(
            packing_list=["Rain shell for the wet days", "Slip-on shoes for temple visits"],
            preparation_checklist=["Reserve Saiho-ji by postcard, two weeks ahead"],
        )
    ]

    await _start(started_runtime, travel_request)
    await _resume(started_runtime, action="approve")

    final = _final(await _state(started_runtime))
    assert final.packing_list == [
        "Rain shell for the wet days",
        "Slip-on shoes for temple visits",
    ]
    assert final.preparation_checklist == ["Reserve Saiho-ji by postcard, two weeks ahead"]
    assert "Kyoto" in final.itinerary.destination


async def test_a_failing_finaliser_still_delivers_the_plan(
    started_runtime: Runtime, travel_request: TravelRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing the packing list is not a reason to lose an approved itinerary."""
    from itinera.agents.schemas import TripPreparation

    original = started_runtime.registry.structured_chain

    def _explode_for_preparation(schema: Any, skip: int = 0) -> Any:
        if schema is TripPreparation:
            raise RuntimeError("the finaliser provider is down")
        return original(schema, skip)

    monkeypatch.setattr(
        started_runtime.registry, "structured_chain", _explode_for_preparation
    )

    await _start(started_runtime, travel_request)
    await _resume(started_runtime, action="approve")

    state = await _state(started_runtime)
    assert state["status"] is PlanStatus.COMPLETED
    final = _final(state)
    assert final.packing_list == []
    assert final.itinerary.duration_days == travel_request.duration_days
