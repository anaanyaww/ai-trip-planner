"""What the agents are asked to emit, as distinct from what the domain stores.

These schemas exist because the domain models carry invariants a language model has no
business being trusted with. ``DraftItinerary`` requires days numbered consecutively
from one; asking a model to maintain that turns a formatting slip into a failed run.
The planner instead emits an ordered list with no numbers at all, and the node assigns
numbers and dates from the request. The invariant is then guaranteed by construction
rather than by the model's care.

The same reasoning applies to research: the brief's location, weather and sources are
recorded by the tools themselves, so the model is never asked to repeat them back.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from itinera.domain.enums import ActivityKind
from itinera.domain.itinerary import Activity, DiningSuggestion
from itinera.domain.research import PointOfInterest, ResearchFinding
from itinera.domain.types import SourceUrl


class ResearchOutput(BaseModel):
    """The research agent's structured findings."""

    attractions: list[PointOfInterest] = Field(
        default_factory=list, description="Candidate places worth scheduling."
    )
    local_tips: list[ResearchFinding] = Field(
        default_factory=list, description="Practical advice for getting around and fitting in."
    )
    safety_notes: list[ResearchFinding] = Field(
        default_factory=list, description="Risks, scams and areas to treat with care."
    )
    seasonal_notes: list[ResearchFinding] = Field(
        default_factory=list,
        description="What these particular dates mean: closures, festivals, crowds.",
    )


class PlannedActivity(BaseModel):
    """One scheduled entry, tagged with the day it belongs to.

    Flat rather than nested inside a day. A nested shape - a list of days each holding
    a list of activities - is what a person would model, and it is also what small
    free-tier models fill least reliably: they return the outer list empty rather than
    admit they cannot build it. A flat list with a day number on each row asks for one
    array of simple rows, which the same models produce dependably.
    """

    day: Annotated[int, Field(ge=1)] = Field(
        description="Which day of the trip this belongs to, counting arrival as 1."
    )
    name: Annotated[str, Field(min_length=1, max_length=160)] = Field(
        description="What the traveller does."
    )
    kind: ActivityKind = Field(
        default=ActivityKind.SIGHTSEEING, description="Classification of the entry."
    )
    duration_hours: Annotated[float, Field(gt=0, le=24)] = Field(
        default=2.0, description="Time on site, including queueing."
    )
    cost_per_person: Annotated[float, Field(ge=0)] = Field(
        default=0.0, description="Spend per traveller. Use 0 for something free."
    )
    neighbourhood: str = Field(default="", description="Area or district.")
    description: str = Field(default="", description="One sentence on why it is worth doing.")
    booking_note: str = Field(default="", description="Anything to arrange in advance.")
    source_urls: list[SourceUrl] = Field(
        default_factory=list, description="Research URLs that support this entry."
    )

    @field_validator("source_urls", mode="before")
    @classmethod
    def _drop_unusable_urls(cls, value: object) -> object:
        """Discard entries that are not URLs rather than failing the whole extraction.

        A model asked for source URLs will occasionally answer "n/a" or a bare page
        title. Rejecting the row would discard a perfectly good activity over its
        citation, so the citation is dropped instead.
        """
        if not isinstance(value, list):
            return value
        return [item for item in value if isinstance(item, str) and item.startswith("http")]

    def to_activity(self) -> Activity:
        """Convert to the domain model the itinerary stores."""
        return Activity(
            name=self.name,
            kind=self.kind,
            duration_hours=self.duration_hours,
            description=self.description,
            neighbourhood=self.neighbourhood or None,
            estimated_cost_per_person=self.cost_per_person,
            booking_note=self.booking_note or None,
            source_urls=self.source_urls,
        )


class PlannedDining(BaseModel):
    """A dining suggestion, tagged with the day it belongs to."""

    day: Annotated[int, Field(ge=1)] = Field(description="Which day of the trip.")
    name: Annotated[str, Field(min_length=1, max_length=160)] = Field(
        description="The restaurant or food experience."
    )
    cuisine: str = Field(default="", description="What kind of food.")
    price_band: Annotated[str, Field(pattern=r"^(budget|mid-range|splurge)$")] = Field(
        default="mid-range", description="One of: budget, mid-range, splurge."
    )
    neighbourhood: str = Field(default="", description="Area or district.")
    rationale: str = Field(default="", description="Why this one.")
    cost_per_person: Annotated[float, Field(ge=0)] = Field(
        default=0.0, description="Spend per traveller."
    )

    def to_suggestion(self) -> DiningSuggestion:
        """Convert to the domain model the itinerary stores."""
        return DiningSuggestion(
            name=self.name,
            cuisine=self.cuisine,
            price_band=self.price_band,
            neighbourhood=self.neighbourhood or None,
            rationale=self.rationale,
            estimated_cost_per_person=self.cost_per_person,
        )


class PlannerOutput(BaseModel):
    """The planner agent's structured itinerary, as flat rows the node assembles."""

    summary: str = Field(default="", description="Two or three sentences framing the trip.")
    day_themes: list[str] = Field(
        default_factory=list,
        description="One short theme per day, in order, starting with day 1.",
    )
    activities: list[PlannedActivity] = Field(
        default_factory=list,
        description="Every scheduled entry across the whole trip, each tagged with its day.",
    )
    dining: list[PlannedDining] = Field(
        default_factory=list, description="Every dining suggestion, each tagged with its day."
    )
    highlights: list[str] = Field(
        default_factory=list, description="The three to five things that define this trip."
    )
    lodging_preference: Annotated[str | None, Field(max_length=200)] = Field(
        default=None, description="The area or style of accommodation the plan assumes."
    )


class RoutingDecision(BaseModel):
    """Which agent a rejection should be routed back to."""

    target: str = Field(
        description="Either 'research' when facts are missing, or 'planner' otherwise."
    )
    reason: str = Field(default="", description="One sentence justifying the choice.")


class TripPreparation(BaseModel):
    """The practical material assembled once a plan is approved."""

    packing_list: list[str] = Field(
        default_factory=list, description="What to bring, specific to these activities."
    )
    preparation_checklist: list[str] = Field(
        default_factory=list, description="What to arrange before departure, in order."
    )
