"""Human-in-the-loop decisions.

The three review actions are modelled as one object with a discriminating ``action``
field rather than three endpoints, because they all resume the same paused workflow
and differ only in the payload they carry.

Each action has different requirements, enforced here so an incoherent decision - a
rejection with no feedback to act on, a modification with nothing to apply - is
rejected at the API boundary rather than confusing the orchestrator.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from itinera.domain.enums import PatchOperation, ReviewActionType
from itinera.domain.itinerary import Activity

_OPERATIONS_REQUIRING_DAY = frozenset(
    {
        PatchOperation.REPLACE_ACTIVITY,
        PatchOperation.ADD_ACTIVITY,
        PatchOperation.REMOVE_ACTIVITY,
        PatchOperation.SET_DAY_NOTES,
    }
)

_OPERATIONS_REQUIRING_INDEX = frozenset(
    {PatchOperation.REPLACE_ACTIVITY, PatchOperation.REMOVE_ACTIVITY}
)

_OPERATIONS_REQUIRING_ACTIVITY = frozenset(
    {PatchOperation.REPLACE_ACTIVITY, PatchOperation.ADD_ACTIVITY}
)

_OPERATIONS_REQUIRING_VALUE = frozenset(
    {PatchOperation.SET_DAY_NOTES, PatchOperation.SET_LODGING_PREFERENCE}
)


class ModificationPatch(BaseModel):
    """A single targeted edit to a draft itinerary.

    Activity positions are one-based to match how the itinerary reads to a person:
    "the second thing on day three" is ``day_number=3, activity_index=2``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: PatchOperation
    day_number: int | None = Field(default=None, ge=1)
    activity_index: int | None = Field(default=None, ge=1)
    activity: Activity | None = None
    value: str | None = None

    @model_validator(mode="after")
    def _validate_operands(self) -> ModificationPatch:
        """Require exactly the operands the chosen operation consumes."""
        if self.operation in _OPERATIONS_REQUIRING_DAY and self.day_number is None:
            raise ValueError(f"{self.operation.value} requires day_number")
        if self.operation in _OPERATIONS_REQUIRING_INDEX and self.activity_index is None:
            raise ValueError(f"{self.operation.value} requires activity_index")
        if self.operation in _OPERATIONS_REQUIRING_ACTIVITY and self.activity is None:
            raise ValueError(f"{self.operation.value} requires an activity payload")
        if self.operation in _OPERATIONS_REQUIRING_VALUE and not (self.value or "").strip():
            raise ValueError(f"{self.operation.value} requires a non-empty value")
        return self

    def describe(self) -> str:
        """Render the patch as a single line for the audit trail."""
        target = f"day {self.day_number}" if self.day_number else "trip"
        if self.activity_index is not None:
            target += f", activity {self.activity_index}"
        return f"{self.operation.value} ({target})"


class ReviewDecision(BaseModel):
    """A user's verdict on a draft itinerary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: ReviewActionType
    feedback: Annotated[str | None, Field(default=None, max_length=2000)] = None
    patches: list[ModificationPatch] = Field(default_factory=list)
    instruction: Annotated[str | None, Field(default=None, max_length=2000)] = None

    @model_validator(mode="after")
    def _validate_payload(self) -> ReviewDecision:
        """Enforce the operands each action needs to be actionable."""
        if self.action is ReviewActionType.REJECT and not (self.feedback or "").strip():
            raise ValueError(
                "reject requires feedback explaining what is wrong, so the workflow "
                "can route the revision to the right agent"
            )
        has_modification = bool(self.patches) or bool((self.instruction or "").strip())
        if self.action is ReviewActionType.MODIFY and not has_modification:
            raise ValueError(
                "modify requires either structured patches or a natural-language "
                "instruction describing the change"
            )
        if self.action is ReviewActionType.APPROVE and self.patches:
            raise ValueError(
                "approve accepts the plan as-is; submit patches with action=modify instead"
            )
        return self

    @property
    def is_structured_modification(self) -> bool:
        """Whether this modification can be applied without an LLM call."""
        return self.action is ReviewActionType.MODIFY and bool(self.patches)

    def audit_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the decision for the audit trail."""
        return {
            "action": self.action.value,
            "feedback": self.feedback,
            "instruction": self.instruction,
            "patches": [patch.describe() for patch in self.patches],
        }
