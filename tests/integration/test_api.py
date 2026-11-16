"""Tests for the HTTP surface.

The workflow's behaviour is covered at the graph level; what matters here is the
contract a client sees. Chiefly the status codes, because those are what a client
branches on: asking for a final plan too early is a 409 rather than a 404, and an
unapplyable patch is a 422 rather than a 200 carrying an error note.

Submission is asynchronous, so most tests drive the run to its pause explicitly rather
than waiting on a background task.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest
from httpx import AsyncClient

from itinera.agents.schemas import PlannedActivity, PlannerOutput
from itinera.domain.enums import PlanStatus
from itinera.domain.travel import TravelRequest
from itinera.llm.providers.fake import FakeChatModel
from itinera.services.planning import PlanningService
from itinera.services.runtime import Runtime

pytestmark = pytest.mark.integration


async def _submit(client: AsyncClient, payload: dict[str, Any], **headers: str) -> str:
    """Submit a plan and return its identifier."""
    response = await client.post("/plan", json=payload, headers=headers)
    assert response.status_code == 202, response.text
    return str(response.json()["plan_id"])


async def _registered(runtime: Runtime, request: TravelRequest) -> str:
    """Register a plan without running it.

    Submitting over HTTP would schedule the background run, and it completes before the
    next request lands - so a state the workflow passes through cannot be observed that
    way. Going through the service directly holds the plan at its initial status.
    """
    plan_id, _ = await PlanningService(runtime).submit(request)
    return plan_id


async def _paused(
    client: AsyncClient, runtime: Runtime, payload: dict[str, Any], request: TravelRequest
) -> str:
    """Submit a plan and run it to the review pause.

    The run is driven here rather than left to the background task so the test does not
    depend on when the event loop gets round to it.
    """
    plan_id = await _submit(client, payload)
    await PlanningService(runtime).run(plan_id, request)
    return plan_id


async def test_submission_is_accepted_and_returns_a_poll_url(
    client: AsyncClient, plan_payload: dict[str, Any]
) -> None:
    response = await client.post("/plan", json=plan_payload)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == PlanStatus.PENDING.value
    assert body["poll_url"] == f"/plan/{body['plan_id']}"


async def test_a_repeated_submission_returns_the_original_plan(
    client: AsyncClient, plan_payload: dict[str, Any]
) -> None:
    """A retried submission must not start a second workflow run."""
    first = await client.post("/plan", json=plan_payload, headers={"Idempotency-Key": "k-1"})
    second = await client.post("/plan", json=plan_payload, headers={"Idempotency-Key": "k-1"})

    assert (first.status_code, second.status_code) == (202, 200)
    assert first.json()["plan_id"] == second.json()["plan_id"]
    assert len((await client.get("/plan")).json()) == 1


async def test_submissions_without_a_key_are_independent(
    client: AsyncClient, plan_payload: dict[str, Any]
) -> None:
    first = await client.post("/plan", json=plan_payload)
    second = await client.post("/plan", json=plan_payload)

    assert first.json()["plan_id"] != second.json()["plan_id"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("destination", "x"),
        ("travelers", 0),
        ("budget_max", -5),
        ("currency", "J1P"),
    ],
)
async def test_malformed_fields_are_rejected(
    client: AsyncClient, plan_payload: dict[str, Any], field: str, value: Any
) -> None:
    response = await client.post("/plan", json={**plan_payload, field: value})

    assert response.status_code == 422


async def test_an_incoherent_date_range_is_a_422_not_a_500(
    client: AsyncClient, plan_payload: dict[str, Any]
) -> None:
    """Cross-field rules live on the domain model, so they raise after parsing."""
    response = await client.post(
        "/plan",
        json={
            **plan_payload,
            "start_date": plan_payload["end_date"],
            "end_date": plan_payload["start_date"],
        },
    )

    assert response.status_code == 422
    assert response.json()["type"] == "validation_error"
    assert "end_date" in response.json()["detail"]


async def test_a_trip_longer_than_the_ceiling_is_rejected(
    client: AsyncClient, plan_payload: dict[str, Any], runtime: Runtime
) -> None:
    """The operational ceiling is enforced at the API, not left to fail in background."""
    start = date.fromisoformat(plan_payload["start_date"])
    over_limit = start + timedelta(days=runtime.settings.max_trip_days + 5)

    response = await client.post(
        "/plan", json={**plan_payload, "end_date": over_limit.isoformat()}
    )

    assert response.status_code == 422
    assert response.json()["type"] == "plan_validation_error"


async def test_status_reflects_the_workflow_stage(
    client: AsyncClient,
    runtime: Runtime,
    plan_payload: dict[str, Any],
    travel_request: TravelRequest,
) -> None:
    plan_id = await _paused(client, runtime, plan_payload, travel_request)

    body = (await client.get(f"/plan/{plan_id}")).json()

    assert body["status"] == PlanStatus.AWAITING_REVIEW.value
    assert body["awaiting_review"] is True
    assert body["draft"] is not None
    assert body["review_request"]["revisions_remaining"] == runtime.settings.max_revisions
    assert [event["event"] for event in body["events"]] == [
        "request_accepted",
        "research_complete",
        "draft_ready",
    ]


async def test_an_unknown_plan_is_a_404(client: AsyncClient) -> None:
    response = await client.get("/plan/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json()["type"] == "plan_not_found"


async def test_reviewing_before_the_pause_is_a_409(
    client: AsyncClient, runtime: Runtime, travel_request: TravelRequest
) -> None:
    """The plan exists; its state is wrong. That is a conflict, not a not-found."""
    plan_id = await _registered(runtime, travel_request)

    response = await client.post(f"/plan/{plan_id}/review", json={"action": "approve"})

    assert response.status_code == 409
    assert response.json()["type"] == "invalid_plan_state"


async def test_the_final_plan_is_unavailable_before_approval(
    client: AsyncClient,
    runtime: Runtime,
    plan_payload: dict[str, Any],
    travel_request: TravelRequest,
) -> None:
    plan_id = await _paused(client, runtime, plan_payload, travel_request)

    response = await client.get(f"/plan/{plan_id}/final")

    assert response.status_code == 409
    assert PlanStatus.AWAITING_REVIEW.value in response.json()["detail"]


async def test_approval_makes_the_final_plan_available(
    client: AsyncClient,
    runtime: Runtime,
    plan_payload: dict[str, Any],
    travel_request: TravelRequest,
) -> None:
    plan_id = await _paused(client, runtime, plan_payload, travel_request)

    review = await client.post(f"/plan/{plan_id}/review", json={"action": "approve"})
    final = await client.get(f"/plan/{plan_id}/final")

    assert review.status_code == 200
    assert review.json()["status"] == PlanStatus.COMPLETED.value
    assert review.json()["final_available"] is True

    assert final.status_code == 200
    body = final.json()
    assert len(body["itinerary"]["days"]) == travel_request.duration_days
    assert body["revision_limit_reached"] is False


async def test_reviewing_a_completed_plan_is_a_409(
    client: AsyncClient,
    runtime: Runtime,
    plan_payload: dict[str, Any],
    travel_request: TravelRequest,
) -> None:
    plan_id = await _paused(client, runtime, plan_payload, travel_request)
    await client.post(f"/plan/{plan_id}/review", json={"action": "approve"})

    response = await client.post(f"/plan/{plan_id}/review", json={"action": "approve"})

    assert response.status_code == 409


async def test_rejection_requires_feedback(
    client: AsyncClient,
    runtime: Runtime,
    plan_payload: dict[str, Any],
    travel_request: TravelRequest,
) -> None:
    """A rejection with nothing to act on cannot be routed anywhere."""
    plan_id = await _paused(client, runtime, plan_payload, travel_request)

    response = await client.post(f"/plan/{plan_id}/review", json={"action": "reject"})

    assert response.status_code == 422


async def test_modification_requires_something_to_apply(
    client: AsyncClient,
    runtime: Runtime,
    plan_payload: dict[str, Any],
    travel_request: TravelRequest,
) -> None:
    plan_id = await _paused(client, runtime, plan_payload, travel_request)

    response = await client.post(f"/plan/{plan_id}/review", json={"action": "modify"})

    assert response.status_code == 422


async def test_a_patch_is_applied_and_the_draft_re_presented(
    client: AsyncClient,
    runtime: Runtime,
    plan_payload: dict[str, Any],
    travel_request: TravelRequest,
    fake_model: FakeChatModel,
) -> None:
    fake_model.structured_responses = [
        PlannerOutput(
            day_themes=["one", "two", "three", "four", "five"],
            activities=[
                PlannedActivity(day=position, name=name)
                for position, name in enumerate(
                    ("one", "two", "three", "four", "five"), start=1
                )
            ],
        )
    ]
    plan_id = await _paused(client, runtime, plan_payload, travel_request)

    response = await client.post(
        f"/plan/{plan_id}/review",
        json={
            "action": "modify",
            "patches": [
                {"operation": "set_day_notes", "day_number": 1, "value": "Arrive late."}
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == PlanStatus.AWAITING_REVIEW.value
    assert body["draft"]["days"][0]["notes"] == "Arrive late."
    assert body["final_available"] is False


async def test_an_unapplyable_patch_is_rejected_without_consuming_the_pause(
    client: AsyncClient,
    runtime: Runtime,
    plan_payload: dict[str, Any],
    travel_request: TravelRequest,
) -> None:
    """The reviewer should be able to correct the patch and simply resubmit."""
    plan_id = await _paused(client, runtime, plan_payload, travel_request)

    response = await client.post(
        f"/plan/{plan_id}/review",
        json={
            "action": "modify",
            "patches": [
                {"operation": "remove_activity", "day_number": 99, "activity_index": 1}
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["type"] == "plan_validation_error"

    still_paused = (await client.get(f"/plan/{plan_id}")).json()
    assert still_paused["status"] == PlanStatus.AWAITING_REVIEW.value

    approved = await client.post(f"/plan/{plan_id}/review", json={"action": "approve"})
    assert approved.status_code == 200


async def test_plans_can_be_listed_and_filtered(
    client: AsyncClient,
    runtime: Runtime,
    plan_payload: dict[str, Any],
    travel_request: TravelRequest,
) -> None:
    paused = await _paused(client, runtime, plan_payload, travel_request)
    pending = await _registered(runtime, travel_request)

    everything = (await client.get("/plan")).json()
    awaiting = (
        await client.get("/plan", params={"status": PlanStatus.AWAITING_REVIEW.value})
    ).json()

    assert {entry["plan_id"] for entry in everything} == {paused, pending}
    assert [entry["plan_id"] for entry in awaiting] == [paused]


async def test_health_and_readiness_report_the_offline_provider(
    client: AsyncClient,
) -> None:
    assert (await client.get("/health")).json() == {"status": "ok"}

    ready = await client.get("/ready")
    body = ready.json()

    assert ready.status_code == 200
    assert body["status"] == "ready"
    assert body["database"] == "ok"
    assert body["warnings"], "an offline runtime must warn that plans are placeholders"


async def test_the_openapi_document_describes_the_review_contract(
    client: AsyncClient,
) -> None:
    """The documented surface is part of the deliverable, so it is worth asserting."""
    schema = (await client.get("/openapi.json")).json()

    assert set(schema["paths"]) >= {
        "/plan",
        "/plan/{plan_id}",
        "/plan/{plan_id}/review",
        "/plan/{plan_id}/final",
    }
    review = schema["paths"]["/plan/{plan_id}/review"]["post"]["responses"]
    assert {"200", "404", "409", "422"} <= set(review)
