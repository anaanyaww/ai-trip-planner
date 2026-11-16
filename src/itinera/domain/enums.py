"""Enumerations shared by the API, the orchestrator and the persistence layer.

These values appear in HTTP responses and are stored in the database, so they are
part of the system's external contract: renaming a member is a breaking change.
"""

from __future__ import annotations

from enum import Enum


class PlanStatus(str, Enum):
    """Lifecycle of a travel plan.

    The status is what ``GET /plan/{id}`` reports, so it must reflect the workflow
    stage precisely rather than collapsing several stages into a generic "running".
    """

    PENDING = "pending"
    RESEARCHING = "researching"
    PLANNING = "planning"
    AWAITING_REVIEW = "awaiting_review"
    REVISING = "revising"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether no further work will happen for a plan in this status."""
        return self in _TERMINAL_STATUSES

    @property
    def accepts_review(self) -> bool:
        """Whether a HITL decision may be submitted while in this status."""
        return self is PlanStatus.AWAITING_REVIEW


_TERMINAL_STATUSES = frozenset(
    {PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED}
)


class ReviewActionType(str, Enum):
    """The three human-in-the-loop decisions the workflow supports."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


class RoutingTarget(str, Enum):
    """Agent a rejection is routed back to.

    The assignment calls for routing to "the appropriate agent(s)", so feedback is
    classified rather than always returned to the planner: a complaint about missing
    facts needs fresh research, whereas a complaint about pacing only needs replanning.
    """

    RESEARCH = "research"
    PLANNER = "planner"


class PatchOperation(str, Enum):
    """Supported structured edits to a draft itinerary.

    Each operation is applied deterministically, without an LLM call, so a targeted
    tweak costs no quota and produces a predictable, testable result.
    """

    REPLACE_ACTIVITY = "replace_activity"
    ADD_ACTIVITY = "add_activity"
    REMOVE_ACTIVITY = "remove_activity"
    SET_DAY_NOTES = "set_day_notes"
    SET_LODGING_PREFERENCE = "set_lodging_preference"


class BudgetCategory(str, Enum):
    """Spending categories the budget allocator distributes across."""

    LODGING = "lodging"
    FOOD = "food"
    ACTIVITIES = "activities"
    TRANSPORT = "transport"
    CONTINGENCY = "contingency"


class ActivityKind(str, Enum):
    """Coarse classification of an itinerary entry, used for pacing heuristics."""

    SIGHTSEEING = "sightseeing"
    MUSEUM = "museum"
    OUTDOOR = "outdoor"
    FOOD = "food"
    NIGHTLIFE = "nightlife"
    SHOPPING = "shopping"
    RELAXATION = "relaxation"
    TRANSIT = "transit"
    EVENT = "event"
