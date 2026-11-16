"""Tests for the day feasibility check.

Each check is exercised in isolation against a day that trips exactly one of them, so
a failure names the rule that broke rather than "the verdict changed".
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from itinera.domain.enums import ActivityKind, BudgetCategory
from itinera.domain.itinerary import BudgetAllocation, BudgetLine
from itinera.domain.research import DailyWeather, WeatherOutlook, WeatherSource
from itinera.tools.budget_allocator import AllocationRecorder
from itinera.tools.day_feasibility import (
    ActivitySketch,
    DayFeasibilityService,
    build_day_feasibility_tool,
)


@pytest.fixture
def allocation() -> BudgetAllocation:
    """Return an envelope allowing 50 per person per day on activities."""
    return BudgetAllocation(
        currency="USD",
        total_budget=2000.0,
        travelers=2,
        duration_days=4,
        lines=[
            BudgetLine(
                category=BudgetCategory.ACTIVITIES,
                total=400.0,
                per_person=200.0,
                per_person_per_day=50.0,
            )
        ],
    )


def _activity(**overrides: object) -> ActivitySketch:
    """Return a well-behaved activity, overridden field by field."""
    defaults = {
        "name": "Nijo Castle",
        "kind": ActivityKind.MUSEUM,
        "duration_hours": 2.5,
        "cost_per_person": 10.0,
        "neighbourhood": "Nakagyo",
    }
    return ActivitySketch(**{**defaults, **overrides})


def _service(settings, start_date, weather=None, allocation=None) -> DayFeasibilityService:
    return DayFeasibilityService(settings, start_date, weather, allocation)


def _reasonable_day() -> list[ActivitySketch]:
    return [
        _activity(name="Nijo Castle"),
        _activity(name="Nishiki Market", kind=ActivityKind.FOOD, cost_per_person=25.0),
    ]


def test_a_reasonable_day_is_feasible(settings, start_date, allocation) -> None:
    verdict = _service(settings, start_date, allocation=allocation).check(1, _reasonable_day())

    assert verdict.is_feasible
    assert verdict.scheduled_hours == pytest.approx(5.0)
    assert verdict.cost_per_person == pytest.approx(35.0)


def test_an_empty_day_is_rejected(settings, start_date) -> None:
    verdict = _service(settings, start_date).check(1, [])

    assert not verdict.is_feasible
    assert "empty" in verdict.issues[0].lower()


def test_an_over_packed_day_is_flagged(settings, start_date, allocation) -> None:
    activities = [
        _activity(name="Fushimi Inari", duration_hours=6.0, cost_per_person=0.0),
        _activity(name="Kiyomizu-dera", duration_hours=5.0, cost_per_person=5.0),
    ]

    verdict = _service(settings, start_date, allocation=allocation).check(1, activities)

    assert any("Over-packed" in issue for issue in verdict.issues)


def test_a_thin_day_is_flagged(settings, start_date, allocation) -> None:
    verdict = _service(settings, start_date, allocation=allocation).check(
        1, [_activity(duration_hours=1.0)]
    )

    assert any("Thin" in issue for issue in verdict.issues)


def test_overspend_is_quantified(settings, start_date, allocation) -> None:
    activities = [_activity(cost_per_person=90.0), _activity(kind=ActivityKind.FOOD)]

    verdict = _service(settings, start_date, allocation=allocation).check(1, activities)

    overspend = next(issue for issue in verdict.issues if "Over budget" in issue)
    assert "50.00 USD" in overspend
    assert "50.00 USD too much" in overspend


def test_missing_allocation_is_a_note_not_a_failure(settings, start_date) -> None:
    verdict = _service(settings, start_date).check(1, _reasonable_day())

    assert verdict.is_feasible
    assert any("budget_allocator" in note for note in verdict.observations)


def test_rain_only_matters_when_the_day_is_spent_outdoors(
    settings, start_date, allocation
) -> None:
    wet = WeatherOutlook(
        source=WeatherSource.FORECAST,
        days=[DailyWeather(day=start_date, temp_min_c=8, temp_max_c=15, precipitation_mm=12.0)],
    )
    service = _service(settings, start_date, weather=wet, allocation=allocation)

    indoors = service.check(1, _reasonable_day())
    outdoors = service.check(
        1,
        [
            _activity(name="Philosopher's Path", kind=ActivityKind.OUTDOOR),
            _activity(name="Arashiyama", kind=ActivityKind.SIGHTSEEING),
        ],
    )

    assert indoors.is_feasible
    assert any("rain expected" in issue for issue in outdoors.issues)
    assert "Philosopher's Path" in outdoors.issues[0]


def test_a_dry_day_passes_with_outdoor_plans(settings, start_date, allocation) -> None:
    dry = WeatherOutlook(
        source=WeatherSource.FORECAST,
        days=[DailyWeather(day=start_date, temp_min_c=12, temp_max_c=22, precipitation_mm=0.0)],
    )

    verdict = _service(settings, start_date, weather=dry, allocation=allocation).check(
        1,
        [
            _activity(name="Philosopher's Path", kind=ActivityKind.OUTDOOR),
            _activity(name="Pontocho dinner", kind=ActivityKind.FOOD),
        ],
    )

    assert verdict.is_feasible
    assert any("22 C" in note for note in verdict.observations)


def test_day_numbers_map_onto_trip_dates(settings, start_date, allocation) -> None:
    outlook = WeatherOutlook(
        source=WeatherSource.FORECAST,
        days=[
            DailyWeather(
                day=start_date + timedelta(days=2),
                temp_min_c=8,
                temp_max_c=15,
                precipitation_mm=20.0,
            )
        ],
    )
    service = _service(settings, start_date, weather=outlook, allocation=allocation)
    outdoor_day = [
        _activity(name="Hike", kind=ActivityKind.OUTDOOR),
        _activity(name="Lunch", kind=ActivityKind.FOOD),
    ]

    assert service.check(1, outdoor_day).is_feasible, "day 1 is not the wet day"
    assert not service.check(3, outdoor_day).is_feasible, "day 3 is the wet day"


def test_cross_town_hopping_is_flagged(settings, start_date, allocation) -> None:
    activities = [
        _activity(name="A", neighbourhood="Gion", duration_hours=1.5),
        _activity(name="B", neighbourhood="Arashiyama", duration_hours=1.5),
        _activity(name="C", neighbourhood="Fushimi", duration_hours=1.5),
        _activity(name="D", neighbourhood="Higashiyama", duration_hours=1.5),
    ]

    verdict = _service(settings, start_date, allocation=allocation).check(1, activities)

    assert any("neighbourhoods" in issue for issue in verdict.issues)


def test_a_missing_meal_is_a_note_not_an_issue(settings, start_date, allocation) -> None:
    verdict = _service(settings, start_date, allocation=allocation).check(
        1, [_activity(duration_hours=4.0)]
    )

    assert verdict.is_feasible
    assert any("No meal scheduled" in note for note in verdict.observations)


def test_a_monotonous_day_is_noted(settings, start_date, allocation) -> None:
    activities = [
        _activity(name=name, kind=ActivityKind.MUSEUM, duration_hours=1.5, cost_per_person=5.0)
        for name in ("A", "B", "C")
    ]

    verdict = _service(settings, start_date, allocation=allocation).check(1, activities)

    assert any("varying the day" in note for note in verdict.observations)


async def test_tool_reads_the_allocation_recorded_after_it_was_built(
    settings, start_date, allocation
) -> None:
    """The allocator runs after the tools are wired, so the envelope must be read late."""
    recorder = AllocationRecorder()
    tool = build_day_feasibility_tool(settings, start_date, allocation_source=recorder)
    payload = {
        "day_number": 1,
        "activities": [{"name": "Museum", "kind": "museum", "duration_hours": 4.0,
                        "cost_per_person": 90.0}],
    }

    before = await tool.ainvoke(payload)
    recorder.record(allocation)
    after = await tool.ainvoke(payload)

    assert "budget_allocator" in before
    assert "Over budget" in after
