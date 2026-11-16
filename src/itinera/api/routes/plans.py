"""The plan endpoints.

The four routes the assignment calls for, plus a listing. The shape follows from the
workflow being asynchronous and interruptible: submission accepts and returns
immediately, the status route reports which stage the workflow reached, the review
route resumes the paused graph, and the final route is available only once a plan has
actually been approved.

Status codes are chosen to be actionable. A review submitted while the workflow is
still researching is a 409 - the plan exists, its state is wrong - not a 404 or a 400.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Header, Query, Response, status

from itinera.api.dependencies import PlanningServiceDep
from itinera.api.schemas.plan import (
    PlanCreatedResponse,
    PlanEventResponse,
    PlanStatusResponse,
    PlanSummaryResponse,
    ProblemResponse,
    ReviewBody,
    ReviewResponse,
    TravelRequestBody,
)
from itinera.core.logging import get_logger
from itinera.domain.enums import PlanStatus, ReviewActionType
from itinera.domain.review import ReviewDecision
from itinera.persistence.models import PlanRecord
from itinera.services.progress import tracker

logger = get_logger(__name__)

router = APIRouter(prefix="/plan", tags=["plans"])

_PROBLEM_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": ProblemResponse, "description": "No such plan"},
    409: {"model": ProblemResponse, "description": "The plan is not in a state allowing this"},
}

_ACTION_MESSAGES = {
    ReviewActionType.APPROVE: "Plan approved. The final plan is available at /plan/{id}/final.",
    ReviewActionType.REJECT: "Feedback accepted. A revised draft is being prepared.",
    ReviewActionType.MODIFY: "Modification applied. Review the updated draft.",
}


def _events(record: PlanRecord) -> list[PlanEventResponse]:
    """Project a record's audit trail into its response shape."""
    return [
        PlanEventResponse(
            event=event.event, detail=event.detail, recorded_at=event.recorded_at
        )
        for event in record.events
    ]


@router.post(
    "",
    response_model=PlanCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a travel request",
    responses={422: {"model": ProblemResponse, "description": "The request is not plannable"}},
)
async def create_plan(
    body: TravelRequestBody,
    background: BackgroundTasks,
    service: PlanningServiceDep,
    response: Response,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            description="Repeat submissions carrying the same key return the original plan.",
        ),
    ] = None,
) -> PlanCreatedResponse:
    """Register a travel request and start planning it in the background.

    Returns 202 rather than 201: the plan resource exists immediately, but the work it
    describes has only been accepted, not completed. Poll ``GET /plan/{id}`` until the
    status reaches ``awaiting_review``.
    """
    request = body.to_domain()
    plan_id, created = await service.submit(request, idempotency_key)

    if created:
        background.add_task(service.run, plan_id, request)
        return PlanCreatedResponse(
            plan_id=plan_id,
            status=PlanStatus.PENDING,
            message="Planning has started. Poll the plan until it is awaiting review.",
            poll_url=f"/plan/{plan_id}",
        )

    response.status_code = status.HTTP_200_OK
    existing = await service.get(plan_id)
    return PlanCreatedResponse(
        plan_id=plan_id,
        status=existing.status,
        message="This request was already submitted; returning the original plan.",
        poll_url=f"/plan/{plan_id}",
    )


@router.get(
    "",
    response_model=list[PlanSummaryResponse],
    summary="List plans",
)
async def list_plans(
    service: PlanningServiceDep,
    plan_status: Annotated[PlanStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PlanSummaryResponse]:
    """Return plans newest-first, optionally filtered by workflow stage."""
    records = await service.list_plans(plan_status, limit, offset)
    return [
        PlanSummaryResponse(
            plan_id=record.plan_id,
            status=record.status,
            destination=record.destination,
            revision_count=record.revision_count,
            updated_at=record.updated_at,
        )
        for record in records
    ]


@router.get(
    "/{plan_id}",
    response_model=PlanStatusResponse,
    summary="Get a plan's status and current draft",
    responses=_PROBLEM_RESPONSES,
)
async def get_plan(plan_id: str, service: PlanningServiceDep) -> PlanStatusResponse:
    """Return the plan's workflow stage, its draft and its audit trail.

    While the workflow is paused, ``review_request`` carries the interrupt's own
    payload - what the graph actually asked a human to look at.
    """
    record = await service.get(plan_id)
    awaiting = record.status.accepts_review

    return PlanStatusResponse(
        plan_id=record.plan_id,
        status=record.status,
        destination=record.destination,
        revision_count=record.revision_count,
        created_at=record.created_at,
        updated_at=record.updated_at,
        awaiting_review=awaiting,
        draft=record.draft,
        review_request=await service.pending_review(plan_id) if awaiting else None,
        error=record.error,
        events=_events(record),
        progress=tracker.snapshot(plan_id),
    )


@router.post(
    "/{plan_id}/review",
    response_model=ReviewResponse,
    summary="Approve, reject or modify a draft",
    responses={
        **_PROBLEM_RESPONSES,
        422: {"model": ProblemResponse, "description": "The decision is not actionable"},
    },
)
async def review_plan(
    plan_id: str, body: ReviewBody, service: PlanningServiceDep
) -> ReviewResponse:
    """Resume the paused workflow with a human decision.

    Approval finalises the plan. A rejection is classified and routed to whichever
    agent can address it. A modification is applied - deterministically when it carries
    patches, by replanning when it carries an instruction - and the result is presented
    for review again, so nothing is finalised without a human seeing it.
    """
    decision = ReviewDecision(**body.model_dump())
    record = await service.review(plan_id, decision)

    return ReviewResponse(
        plan_id=record.plan_id,
        status=record.status,
        action=decision.action,
        revision_count=record.revision_count,
        message=_ACTION_MESSAGES[decision.action],
        draft=record.draft,
        final_available=record.final is not None,
    )


@router.get(
    "/{plan_id}/final",
    summary="Retrieve the finalised plan",
    responses=_PROBLEM_RESPONSES,
)
async def get_final_plan(plan_id: str, service: PlanningServiceDep) -> dict[str, Any]:
    """Return the approved plan with its packing and preparation material.

    Available only after approval; asking earlier is a 409 naming the current status.
    """
    final = await service.get_final(plan_id)
    return final.model_dump(mode="json")
