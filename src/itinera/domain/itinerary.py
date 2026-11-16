"""The itinerary: the planner agent's output and the object the user reviews.

Two properties of this module matter to the workflow as a whole.

First, the structures are addressable. A day has a number, activities within it have
positions, and lodging is a named field. That is what makes the "modify" review action
a deterministic patch rather than another round of generation - the user points at
``day 3, activity 2`` and the system edits exactly that.

Second, costs roll up rather than being asserted. A day's cost is the sum of its
activities, and the trip's cost is the sum of its days, so an itinerary cannot claim
to fit a budget it does not fit.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from itinera.domain.enums import ActivityKind, BudgetCategory
from itinera.domain.research import DailyWeather
from itinera.domain.types import SourceUrl


class BudgetLine(BaseModel):
    """One category's share of the budget."""

    model_config = ConfigDict(frozen=True)

    category: BudgetCategory
    total: Annotated[float, Field(ge=0)]
    per_person: Annotated[float, Field(ge=0)]
    per_person_per_day: Annotated[float, Field(ge=0)]


class BudgetAllocation(BaseModel):
    """The spending envelope the planner must build within.

    Produced deterministically by the budget allocator tool from the request's budget
    range, party size and trip length. When the destination trades in a different
    currency, the converted view is carried alongside the original so the plan can
    quote local prices without losing the figure the user actually supplied.
    """

    currency: str
    total_budget: Annotated[float, Field(ge=0)]
    travelers: Annotated[int, Field(ge=1)]
    duration_days: Annotated[int, Field(ge=1)]
    lines: list[BudgetLine] = Field(default_factory=list)
    local_currency: str | None = None
    exchange_rate: float | None = Field(default=None, gt=0)
    fx_rate_date: date | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def daily_envelope_per_person(self) -> float:
        """Spend available per traveller per day across all categories."""
        return round(self.total_budget / (self.travelers * self.duration_days), 2)

    @property
    def activities_per_person_per_day(self) -> float:
        """Daily activity spend per traveller, the figure the planner schedules against."""
        for line in self.lines:
            if line.category is BudgetCategory.ACTIVITIES:
                return line.per_person_per_day
        return 0.0

    def to_local(self, amount: float) -> float | None:
        """Convert ``amount`` into the destination currency, if a rate is known."""
        if self.exchange_rate is None:
            return None
        return round(amount * self.exchange_rate, 2)


class Activity(BaseModel):
    """A single scheduled entry within a day."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: Annotated[str, Field(min_length=1, max_length=160)]
    kind: ActivityKind = ActivityKind.SIGHTSEEING
    start_time: time | None = None
    duration_hours: Annotated[float, Field(gt=0, le=24)] = 2.0
    description: str = ""
    neighbourhood: str | None = None
    estimated_cost_per_person: Annotated[float, Field(ge=0)] = 0.0
    booking_note: str | None = None
    source_urls: list[SourceUrl] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _reject_blank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("activity name must not be blank")
        return value.strip()


class DiningSuggestion(BaseModel):
    """A restaurant or food experience recommended for a given day."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: Annotated[str, Field(min_length=1, max_length=160)]
    cuisine: str = ""
    price_band: Annotated[str, Field(pattern=r"^(budget|mid-range|splurge)$")] = "mid-range"
    neighbourhood: str | None = None
    rationale: str = ""
    estimated_cost_per_person: Annotated[float, Field(ge=0)] = 0.0
    source_url: SourceUrl | None = None


class LocalEvent(BaseModel):
    """A dated happening that falls inside the travel window."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: Annotated[str, Field(min_length=1, max_length=160)]
    event_date: date | None = None
    venue: str | None = None
    description: str = ""
    source_url: SourceUrl | None = None


class DayPlan(BaseModel):
    """One day of the trip."""

    day_number: Annotated[int, Field(ge=1)]
    day_date: date
    theme: str = ""
    activities: list[Activity] = Field(default_factory=list)
    dining: list[DiningSuggestion] = Field(default_factory=list)
    events: list[LocalEvent] = Field(default_factory=list)
    notes: str = ""
    weather: DailyWeather | None = None

    @property
    def estimated_cost_per_person(self) -> float:
        """Total scheduled spend per traveller for this day."""
        activities = sum(item.estimated_cost_per_person for item in self.activities)
        dining = sum(item.estimated_cost_per_person for item in self.dining)
        return round(activities + dining, 2)

    @property
    def scheduled_hours(self) -> float:
        """Total hours of scheduled activity, used to detect an over-packed day."""
        return round(sum(item.duration_hours for item in self.activities), 2)

    @property
    def is_empty(self) -> bool:
        """Whether the day has nothing scheduled at all."""
        return not self.activities and not self.dining


class DraftItinerary(BaseModel):
    """The plan presented to the user for review.

    Carries the revision counter so both the API and the audit trail can show how
    many rounds of feedback shaped what the user is looking at.
    """

    destination: str
    start_date: date
    end_date: date
    travelers: Annotated[int, Field(ge=1)]
    summary: str = ""
    days: list[DayPlan] = Field(default_factory=list)
    budget: BudgetAllocation | None = None
    lodging_preference: Annotated[str | None, Field(max_length=200)] = None
    highlights: list[str] = Field(default_factory=list)
    revision: Annotated[int, Field(ge=0)] = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("days")
    @classmethod
    def _validate_day_numbering(cls, value: list[DayPlan]) -> list[DayPlan]:
        """Reject gaps and duplicates, since patches address days by number."""
        expected = list(range(1, len(value) + 1))
        actual = [day.day_number for day in value]
        if actual != expected:
            raise ValueError(
                f"days must be numbered consecutively from 1; got {actual}"
            )
        return value

    @property
    def total_estimated_cost(self) -> float:
        """Whole-party cost of everything scheduled across the trip."""
        per_person = sum(day.estimated_cost_per_person for day in self.days)
        return round(per_person * self.travelers, 2)

    @property
    def duration_days(self) -> int:
        """Number of days the itinerary covers."""
        return len(self.days)

    def day(self, day_number: int) -> DayPlan | None:
        """Return the day with ``day_number``, or ``None`` when it does not exist."""
        for entry in self.days:
            if entry.day_number == day_number:
                return entry
        return None

    def fits_budget(self) -> bool:
        """Whether scheduled spend stays within the allocated envelope.

        Compared against the activities-and-food share rather than the whole budget,
        since lodging and transport are not itemised as activities.
        """
        if self.budget is None:
            return True
        allowance = self.budget.activities_per_person_per_day * self.duration_days
        spend_per_person = sum(day.estimated_cost_per_person for day in self.days)
        return spend_per_person <= allowance


class FinalPlan(BaseModel):
    """The approved itinerary plus the trip-preparation material.

    The packing and preparation lists are assembled after approval rather than being
    drafted with the itinerary: they depend on the weather outlook and on the
    activities that actually survived review.
    """

    itinerary: DraftItinerary
    packing_list: list[str] = Field(default_factory=list)
    preparation_checklist: list[str] = Field(default_factory=list)
    weather_summary: str = ""
    sources: list[str] = Field(default_factory=list)
    revision_count: Annotated[int, Field(ge=0)] = 0
    revision_limit_reached: bool = False
    approved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_estimated_cost(self) -> float:
        """Whole-party cost of the approved itinerary."""
        return self.itinerary.total_estimated_cost
