"""Day feasibility: the planner agent's second tool.

The budget allocator constrains what a trip may cost; this tool constrains what a day
can physically contain. Both exist for the same reason - a language model asked to
produce a plausible itinerary will produce a plausible-looking one, scheduling eleven
hours of sightseeing across four neighbourhoods on a day forecast to rain.

The checks are deterministic and run against facts the workflow already holds: the
weather outlook the research agent gathered, and the envelope the allocator produced.
Nothing here is judged by a model, so the same day always gets the same verdict, and
the thresholds that define "over-packed" are configuration rather than opinion baked
into code.

The tool reports rather than rejects. It returns findings the planner is expected to
act on, because a day that trips one check is often the right day once the planner
adds an indoor fallback or drops an activity - and refusing outright would leave the
agent with nothing to do but guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Annotated

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from itinera.core.config import Settings
from itinera.core.logging import get_logger
from itinera.domain.enums import ActivityKind
from itinera.domain.itinerary import BudgetAllocation
from itinera.domain.research import DailyWeather, WeatherOutlook
from itinera.tools.budget_allocator import AllocationRecorder

logger = get_logger(__name__)

_TOOL_NAME = "day_feasibility_check"

_WEATHER_EXPOSED = frozenset({ActivityKind.OUTDOOR, ActivityKind.SIGHTSEEING})

_MEAL_KINDS = frozenset({ActivityKind.FOOD})


class ActivitySketch(BaseModel):
    """A proposed activity, described only as far as the checks need it."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: Annotated[str, Field(min_length=1, max_length=160)]
    kind: ActivityKind = Field(
        default=ActivityKind.SIGHTSEEING,
        description="Classification used for weather and variety checks.",
    )
    duration_hours: Annotated[float, Field(gt=0, le=24)] = Field(
        default=2.0, description="Time on site including queueing, in hours."
    )
    cost_per_person: Annotated[float, Field(ge=0)] = Field(
        default=0.0, description="Admission or spend per traveller, in the trip currency."
    )
    neighbourhood: str | None = Field(
        default=None, description="Area or district, used to detect cross-town hopping."
    )


class DayFeasibilityInput(BaseModel):
    """Arguments accepted by the feasibility check."""

    day_number: Annotated[int, Field(ge=1)] = Field(
        description="Which day of the trip this is, counting the arrival day as 1."
    )
    activities: list[ActivitySketch] = Field(
        description="Everything currently scheduled for the day, in order."
    )


@dataclass
class DayVerdict:
    """The outcome of checking one day."""

    day_number: int
    day_date: date | None
    scheduled_hours: float
    cost_per_person: float
    issues: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        """Whether the day passed every check."""
        return not self.issues


class DayFeasibilityService:
    """Applies the pacing, budget and weather checks to a proposed day."""

    def __init__(
        self,
        settings: Settings,
        start_date: date,
        weather: WeatherOutlook | None = None,
        allocation: BudgetAllocation | None = None,
    ) -> None:
        """Store the thresholds and the run facts the checks are made against.

        Args:
            settings: Supplies the pacing thresholds.
            start_date: First day of the trip, used to map day numbers to dates.
            weather: Outlook from the research agent, when one was obtained.
            allocation: Envelope from the budget allocator, when it has run.
        """
        self._settings = settings
        self._start_date = start_date
        self._weather = weather
        self._allocation = allocation

    def _date_for(self, day_number: int) -> date:
        """Return the calendar date day ``day_number`` falls on."""
        return self._start_date + timedelta(days=day_number - 1)

    def _weather_for(self, day: date) -> DailyWeather | None:
        """Return the weather entry for ``day``, or ``None`` when not covered."""
        if self._weather is None:
            return None
        for entry in self._weather.days:
            if entry.day == day:
                return entry
        return None

    def _check_pacing(self, verdict: DayVerdict) -> None:
        """Flag days that are exhausting or too thin to be worth the traveller's time."""
        hours = verdict.scheduled_hours
        ceiling = self._settings.day_max_scheduled_hours
        if hours > ceiling:
            verdict.issues.append(
                f"Over-packed: {hours:.1f}h scheduled against a {ceiling:.1f}h ceiling. "
                "Drop or shorten an activity."
            )
        elif hours < self._settings.day_min_scheduled_hours:
            verdict.issues.append(
                f"Thin: only {hours:.1f}h scheduled against a "
                f"{self._settings.day_min_scheduled_hours:.1f}h floor. Add something."
            )

    def _check_budget(self, verdict: DayVerdict) -> None:
        """Flag days that spend beyond the allocated daily activity envelope."""
        if self._allocation is None:
            verdict.observations.append(
                "No budget allocation available yet; spend was not checked. "
                "Call budget_allocator first."
            )
            return

        envelope = self._allocation.activities_per_person_per_day
        currency = self._allocation.currency
        if verdict.cost_per_person > envelope:
            overspend = verdict.cost_per_person - envelope
            verdict.issues.append(
                f"Over budget: {verdict.cost_per_person:.2f} {currency} per person "
                f"against a {envelope:.2f} {currency} daily envelope, "
                f"{overspend:.2f} {currency} too much."
            )
        else:
            headroom = envelope - verdict.cost_per_person
            verdict.observations.append(
                f"Within budget, {headroom:.2f} {currency} per person unspent."
            )

    def _check_weather(self, verdict: DayVerdict, activities: list[ActivitySketch]) -> None:
        """Flag outdoor plans on a wet day and note the temperature extremes."""
        if verdict.day_date is None:
            return

        entry = self._weather_for(verdict.day_date)
        if entry is None:
            verdict.observations.append("No weather data covers this day.")
            return

        exposed = [item.name for item in activities if item.kind in _WEATHER_EXPOSED]
        if entry.is_wet and exposed:
            verdict.issues.append(
                f"{entry.precipitation_mm:.0f}mm of rain expected and {len(exposed)} "
                f"outdoor item(s) scheduled ({', '.join(exposed)}). Add an indoor "
                "alternative or move them to a drier day."
            )
        verdict.observations.append(
            f"Weather: {entry.temp_min_c:.0f} to {entry.temp_max_c:.0f} C, "
            f"{entry.precipitation_mm:.0f}mm precipitation."
        )

    def _check_geography(self, verdict: DayVerdict, activities: list[ActivitySketch]) -> None:
        """Flag days that criss-cross the city more than the transit budget allows."""
        areas = {
            item.neighbourhood.strip()
            for item in activities
            if item.neighbourhood and item.neighbourhood.strip()
        }
        ceiling = self._settings.day_max_neighbourhoods
        if len(areas) > ceiling:
            verdict.issues.append(
                f"Spread across {len(areas)} neighbourhoods ({', '.join(sorted(areas))}) "
                f"against a ceiling of {ceiling}. Group the day geographically."
            )

    @staticmethod
    def _check_composition(verdict: DayVerdict, activities: list[ActivitySketch]) -> None:
        """Flag a day with no meal and a day that repeats a single kind of activity."""
        kinds = {item.kind for item in activities}
        if not kinds & _MEAL_KINDS:
            verdict.observations.append(
                "No meal scheduled. Either add one or leave a gap for it."
            )
        if len(activities) > 2 and len(kinds) == 1:
            only = next(iter(kinds))
            verdict.observations.append(
                f"Every activity is {only.value}; varying the day reads better."
            )

    def check(self, day_number: int, activities: list[ActivitySketch]) -> DayVerdict:
        """Run every check against a proposed day.

        Args:
            day_number: Which day of the trip is being checked.
            activities: Everything currently scheduled for it.

        Returns:
            The verdict, carrying blocking issues and non-blocking observations.
        """
        verdict = DayVerdict(
            day_number=day_number,
            day_date=self._date_for(day_number),
            scheduled_hours=round(sum(item.duration_hours for item in activities), 2),
            cost_per_person=round(sum(item.cost_per_person for item in activities), 2),
        )

        if not activities:
            verdict.issues.append("The day is empty. Schedule at least one activity.")
            return verdict

        self._check_pacing(verdict)
        self._check_budget(verdict)
        self._check_weather(verdict, activities)
        self._check_geography(verdict, activities)
        self._check_composition(verdict, activities)
        return verdict


def _format(verdict: DayVerdict) -> str:
    """Render the verdict as prose the planner can act on."""
    header = f"Day {verdict.day_number}"
    if verdict.day_date is not None:
        header += f" ({verdict.day_date.isoformat()})"
    header += (
        f": {verdict.scheduled_hours:.1f}h scheduled, "
        f"{verdict.cost_per_person:.2f} per person."
    )

    lines = [header]
    if verdict.is_feasible:
        lines.append("Feasible as planned.")
    else:
        lines.append("Revise this day before continuing:")
        lines.extend(f"  - {issue}" for issue in verdict.issues)

    if verdict.observations:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in verdict.observations)
    return "\n".join(lines)


def build_day_feasibility_tool(
    settings: Settings,
    start_date: date,
    weather: WeatherOutlook | None = None,
    allocation_source: AllocationRecorder | None = None,
) -> BaseTool:
    """Build the feasibility check bound to a run's weather and budget.

    Args:
        settings: Supplies the pacing thresholds.
        start_date: First day of the trip.
        weather: Outlook from the research agent, when one was obtained.
        allocation_source: Recorder the budget allocator writes to. It is read at
            call time rather than bound up front, so the check sees the envelope even
            though the allocator runs after the tools are built.

    Returns:
        A LangChain tool the planner agent can call.
    """

    def _check(day_number: int, activities: list[ActivitySketch]) -> str:
        """Check one proposed day for pacing, budget and weather problems."""
        allocation = allocation_source.allocation if allocation_source else None
        service = DayFeasibilityService(settings, start_date, weather, allocation)
        verdict = service.check(day_number, activities)

        logger.info(
            "feasibility.checked",
            day=day_number,
            feasible=verdict.is_feasible,
            issues=len(verdict.issues),
        )
        return _format(verdict)

    return StructuredTool.from_function(
        func=_check,
        name=_TOOL_NAME,
        description=(
            "Check a proposed day against the trip's budget envelope, the weather "
            "forecast and sensible pacing limits before committing to it. Returns the "
            "problems to fix, such as an over-packed schedule, overspend, or outdoor "
            "plans on a wet day. Call this for each day as you draft it."
        ),
        args_schema=DayFeasibilityInput,
    )
