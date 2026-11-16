"""Rendering domain objects into prompt context.

Passing a Pydantic model's JSON dump into a prompt spends tokens on field names,
nulls and punctuation that carry no meaning for the reader. These renderers emit the
same information as compact prose, which both costs less and is easier for a small
model to follow.

They are also what the API's human-readable views are built from, so the text the
planner reads and the text a reviewer reads come from one implementation.
"""

from __future__ import annotations

from itinera.domain.itinerary import DayPlan, DraftItinerary
from itinera.domain.research import ResearchBrief, ResearchFinding, WeatherOutlook

_MAX_SOURCES_IN_PROMPT = 12
_MAX_ATTRACTIONS_IN_PROMPT = 20
_MAX_FINDINGS_PER_TOPIC = 8
_MAX_FINDING_CHARS = 400


def _findings(title: str, findings: list[ResearchFinding]) -> list[str]:
    """Render one topical group of findings, or nothing when the group is empty."""
    if not findings:
        return []
    lines = [f"### {title}"]
    lines.extend(
        f"- {finding.topic}: {finding.content[:_MAX_FINDING_CHARS]}"
        for finding in findings[:_MAX_FINDINGS_PER_TOPIC]
    )
    if len(findings) > _MAX_FINDINGS_PER_TOPIC:
        lines.append(f"- ({len(findings) - _MAX_FINDINGS_PER_TOPIC} further findings omitted)")
    return lines


def render_weather(outlook: WeatherOutlook | None) -> str:
    """Render a weather outlook, making the forecast/normals distinction explicit."""
    if outlook is None or not outlook.days:
        return "No weather information is available for these dates."

    lines = [outlook.summary]
    if outlook.average_high_c is not None:
        lines.append(
            f"Average high {outlook.average_high_c} C, average low {outlook.average_low_c} C."
        )
    wet = outlook.wet_days
    lines.append(
        "Expect rain on: " + ", ".join(day.isoformat() for day in wet)
        if wet
        else "No notably wet days expected."
    )
    return "\n".join(lines)


def render_brief(brief: ResearchBrief) -> str:
    """Render a research brief as the planner's working context."""
    lines: list[str] = []

    if brief.location is not None:
        lines.append(f"Location: {brief.location.label}")
    lines.append(render_weather(brief.weather))

    if brief.attractions:
        lines.append("\n### Candidate attractions")
        for place in brief.attractions[:_MAX_ATTRACTIONS_IN_PROMPT]:
            detail = [place.name, f"{place.kind.value}", f"{place.typical_duration_hours}h"]
            if place.neighbourhood:
                detail.append(place.neighbourhood)
            if place.estimated_cost_per_person is not None:
                detail.append(f"~{place.estimated_cost_per_person:.0f} per person")
            lines.append(f"- {' | '.join(detail)}")
            if place.description:
                lines.append(f"  {place.description}")

    lines.extend(_findings("Local tips", brief.local_tips))
    lines.extend(_findings("Safety", brief.safety_notes))
    lines.extend(_findings("Seasonal considerations", brief.seasonal_notes))

    if brief.sources:
        lines.append("\n### Sources")
        lines.extend(
            f"- {result.as_citation()}"
            for result in brief.sources[:_MAX_SOURCES_IN_PROMPT]
        )

    if not brief.is_usable:
        lines.append(
            "\nThis brief is thin. Say so in the plan's summary rather than filling the "
            "gaps from memory."
        )
    return "\n".join(lines)


def render_day(day: DayPlan) -> str:
    """Render one day of an itinerary."""
    header = f"Day {day.day_number} ({day.day_date.isoformat()})"
    if day.theme:
        header += f" - {day.theme}"

    lines = [header]
    for position, activity in enumerate(day.activities, start=1):
        detail = f"  {position}. {activity.name} ({activity.duration_hours}h"
        if activity.estimated_cost_per_person:
            detail += f", {activity.estimated_cost_per_person:.0f} per person"
        if activity.neighbourhood:
            detail += f", {activity.neighbourhood}"
        lines.append(detail + ")")

    for suggestion in day.dining:
        lines.append(f"  Dining: {suggestion.name} ({suggestion.price_band})")
    for event in day.events:
        lines.append(f"  Event: {event.name}")
    if day.notes:
        lines.append(f"  Notes: {day.notes}")
    return "\n".join(lines)


def render_itinerary(itinerary: DraftItinerary) -> str:
    """Render a draft itinerary as prose, for prompts and for review."""
    lines = [
        f"{itinerary.destination}, {itinerary.start_date.isoformat()} to "
        f"{itinerary.end_date.isoformat()}, {itinerary.travelers} traveller(s)."
    ]
    if itinerary.summary:
        lines.append(itinerary.summary)
    if itinerary.lodging_preference:
        lines.append(f"Lodging: {itinerary.lodging_preference}")

    lines.extend(render_day(day) for day in itinerary.days)

    if itinerary.budget is not None:
        lines.append(
            f"Estimated total: {itinerary.total_estimated_cost:.0f} "
            f"{itinerary.budget.currency} for the whole party."
        )
    return "\n\n".join(lines)
