"""Deterministic application of modification patches to a draft itinerary.

This is the code path behind the "modify" review action. It runs no model and makes
no network call: a patch either addresses something that exists and is applied
exactly, or it addresses something that does not and is rejected with a message
naming what was actually available.

Patches are applied to a copy and swapped in only once every one of them has
succeeded, so a batch containing a bad patch leaves the itinerary untouched rather
than half-edited.
"""

from __future__ import annotations

from collections.abc import Sequence

from itinera.core.exceptions import PlanValidationError
from itinera.domain.enums import PatchOperation
from itinera.domain.itinerary import DayPlan, DraftItinerary
from itinera.domain.review import ModificationPatch


def _require_day(itinerary: DraftItinerary, day_number: int | None) -> DayPlan:
    """Return the addressed day, or explain which days exist.

    Raises:
        PlanValidationError: If the itinerary has no such day.
    """
    if day_number is None:  # pragma: no cover - guarded by ModificationPatch validation
        raise PlanValidationError("patch is missing day_number")

    day = itinerary.day(day_number)
    if day is None:
        raise PlanValidationError(
            f"Day {day_number} does not exist; this itinerary covers days "
            f"1 to {itinerary.duration_days}.",
            day_number=day_number,
            available_days=itinerary.duration_days,
        )
    return day


def _require_activity_position(day: DayPlan, activity_index: int | None) -> int:
    """Return the zero-based position of a one-based activity index.

    Raises:
        PlanValidationError: If the day has no activity at that position.
    """
    if activity_index is None:  # pragma: no cover - guarded by ModificationPatch validation
        raise PlanValidationError("patch is missing activity_index")

    if not 1 <= activity_index <= len(day.activities):
        available = (
            f"1 to {len(day.activities)}" if day.activities else "none - the day is empty"
        )
        raise PlanValidationError(
            f"Day {day.day_number} has no activity {activity_index}; "
            f"available positions: {available}.",
            day_number=day.day_number,
            activity_index=activity_index,
            available_activities=len(day.activities),
        )
    return activity_index - 1


def _apply_one(itinerary: DraftItinerary, patch: ModificationPatch) -> None:
    """Apply a single patch in place.

    Raises:
        PlanValidationError: If the patch addresses something that does not exist.
    """
    if patch.operation is PatchOperation.SET_LODGING_PREFERENCE:
        itinerary.lodging_preference = patch.value
        return

    day = _require_day(itinerary, patch.day_number)

    if patch.operation is PatchOperation.SET_DAY_NOTES:
        day.notes = patch.value or ""

    elif patch.operation is PatchOperation.REPLACE_ACTIVITY:
        position = _require_activity_position(day, patch.activity_index)
        assert patch.activity is not None  # noqa: S101 - guaranteed by patch validation
        day.activities[position] = patch.activity

    elif patch.operation is PatchOperation.REMOVE_ACTIVITY:
        position = _require_activity_position(day, patch.activity_index)
        day.activities.pop(position)

    elif patch.operation is PatchOperation.ADD_ACTIVITY:
        assert patch.activity is not None  # noqa: S101 - guaranteed by patch validation
        if patch.activity_index is None:
            day.activities.append(patch.activity)
        else:
            position = min(patch.activity_index - 1, len(day.activities))
            day.activities.insert(position, patch.activity)


def apply_patches(
    itinerary: DraftItinerary, patches: Sequence[ModificationPatch]
) -> DraftItinerary:
    """Return a copy of ``itinerary`` with every patch applied.

    Args:
        itinerary: The draft under review.
        patches: Edits to apply, in order. Later patches see the effect of earlier
            ones, so removing two activities from the same day must account for the
            renumbering the first removal causes.

    Returns:
        A new itinerary with the edits applied and its revision counter advanced.

    Raises:
        PlanValidationError: If any patch addresses a day or activity that does not
            exist. The original itinerary is left unmodified in that case.
    """
    if not patches:
        return itinerary

    working = itinerary.model_copy(deep=True)
    for patch in patches:
        _apply_one(working, patch)

    working.revision = itinerary.revision + 1
    return working
