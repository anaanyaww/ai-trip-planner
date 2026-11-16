"""Request and response bodies for the plan endpoints.

The API's models are deliberately separate from the domain's. ``TravelRequest`` is
frozen and validates against configuration; a request body needs neither and needs an
OpenAPI example instead. Keeping them apart means the wire format can gain a field, or
rename one, without the workflow's models moving.

Responses carry the workflow stage verbatim rather than collapsing it into a generic
"running", because the assignment asks the status to reflect where the workflow is,
and because "researching" and "awaiting_review" call for very different client
behaviour.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from itinera.domain.enums import PlanStatus, ReviewActionType
from itinera.domain.review import ModificationPatch
from itinera.domain.travel import TravelRequest


class TravelRequestBody(BaseModel):
    """A travel request as submitted to ``POST /plan``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "destination": "Kyoto, Japan",
                "start_date": "2026-11-10",
                "end_date": "2026-11-14",
                "budget_min": 1800,
                "budget_max": 2600,
                "currency": "USD",
                "interests": ["temples", "food", "autumn foliage"],
                "travelers": 2,
            }
        },
    )

    destination: Annotated[str, Field(min_length=2, max_length=120)]
    start_date: date
    end_date: date
    budget_min: Annotated[float, Field(ge=0)]
    budget_max: Annotated[float, Field(gt=0)]
    currency: Annotated[str, Field(min_length=3, max_length=3)] = "USD"
    interests: list[str] = Field(default_factory=list)
    travelers: Annotated[int, Field(ge=1)] = 1

    def to_domain(self) -> TravelRequest:
        """Convert the body into the domain request.

        Raises:
            ValidationError: If the dates or budget range are incoherent. FastAPI
                translates this into a 422 with the offending field named.
        """
        return TravelRequest(**self.model_dump())


class ReviewBody(BaseModel):
    """A human-in-the-loop decision submitted to ``POST /plan/{id}/review``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"action": "approve"},
                {
                    "action": "reject",
                    "feedback": "Day 2 is exhausting and nothing covers the food markets.",
                },
                {
                    "action": "modify",
                    "patches": [
                        {
                            "operation": "remove_activity",
                            "day_number": 3,
                            "activity_index": 2,
                        }
                    ],
                },
                {
                    "action": "modify",
                    "instruction": "Swap the day 4 museum for something outdoors.",
                },
            ]
        },
    )

    action: ReviewActionType
    feedback: Annotated[str | None, Field(default=None, max_length=2000)] = None
    patches: list[ModificationPatch] = Field(default_factory=list)
    instruction: Annotated[str | None, Field(default=None, max_length=2000)] = None


class PlanCreatedResponse(BaseModel):
    """The acknowledgement returned by ``POST /plan``."""

    plan_id: str
    status: PlanStatus
    message: str
    poll_url: str


class PlanEventResponse(BaseModel):
    """One entry in a plan's audit trail."""

    event: str
    detail: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime


class PlanStatusResponse(BaseModel):
    """The current state of a plan, as returned by ``GET /plan/{id}``."""

    plan_id: str
    status: PlanStatus
    destination: str
    revision_count: int
    created_at: datetime
    updated_at: datetime
    awaiting_review: bool = Field(
        description="Whether the workflow is paused and a review would be accepted."
    )
    draft: dict[str, Any] | None = Field(
        default=None, description="The itinerary under review, once one exists."
    )
    review_request: dict[str, Any] | None = Field(
        default=None,
        description="What the paused workflow asked a human to look at, including a "
        "rendered view of the draft and how many revisions remain.",
    )
    error: str | None = None
    events: list[PlanEventResponse] = Field(default_factory=list)
    progress: dict[str, Any] | None = Field(
        default=None,
        description="Live detail for a run still in flight: seconds elapsed and the "
        "most recent things the agents did. Absent once the run leaves this process, "
        "which is why it supplements the audit trail rather than replacing it.",
    )


class ReviewResponse(BaseModel):
    """The outcome of submitting a review."""

    plan_id: str
    status: PlanStatus
    action: ReviewActionType
    revision_count: int
    message: str
    draft: dict[str, Any] | None = None
    final_available: bool = False


class PlanSummaryResponse(BaseModel):
    """A plan as it appears in a listing."""

    plan_id: str
    status: PlanStatus
    destination: str
    revision_count: int
    updated_at: datetime


class ProblemResponse(BaseModel):
    """An error, in the shape of RFC 7807.

    A machine-readable ``type`` and a human-readable ``detail`` mean a client can
    branch on the failure without parsing prose.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "invalid_plan_state",
                "title": "Invalid Plan State",
                "status": 409,
                "detail": "Plan 0f3c is researching and is not awaiting review.",
            }
        }
    )

    type: str
    title: str
    status: int
    detail: str
