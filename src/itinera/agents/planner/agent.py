"""The itinerary planner agent.

The agent emits flat rows - every activity and meal tagged with the day it belongs to -
and this module groups them into days, assigning numbers, dates and weather from the
travel request. Two things follow from that division. The itinerary's invariants hold
by construction, because the model never numbers a day. And the shape the model is
asked for is a single list of simple rows rather than a list of days each holding a
list of activities, which is what free-tier models fill reliably; asked for the nested
form they tend to return the outer list empty.

The budget allocation is likewise taken from the allocator's recorder rather than from
the model's rendering of it, so the figure attached to the plan is the one the
arithmetic produced.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from itinera.agents.base import ToolCallingAgent
from itinera.agents.rendering import render_brief, render_itinerary
from itinera.agents.schemas import PlannerOutput
from itinera.core.config import Settings
from itinera.core.logging import get_logger
from itinera.domain.itinerary import Activity, DayPlan, DiningSuggestion, DraftItinerary
from itinera.domain.research import ResearchBrief
from itinera.domain.travel import TravelRequest
from itinera.llm.registry import ModelRegistry
from itinera.prompts.library import PromptLibrary
from itinera.tools.base import ToolContext
from itinera.tools.budget_allocator import AllocationRecorder, build_budget_allocator_tool
from itinera.tools.day_feasibility import build_day_feasibility_tool

logger = get_logger(__name__)

_AGENT_NAME = "planner"


class PlannerAgent:
    """Turns a research brief into a reviewable day-by-day itinerary."""

    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        prompts: PromptLibrary,
        context: ToolContext,
    ) -> None:
        """Wire the agent to its models, prompts and tool dependencies."""
        self._settings = settings
        self._registry = registry
        self._prompts = prompts
        self._context = context

    async def run(
        self,
        request: TravelRequest,
        brief: ResearchBrief,
        previous: DraftItinerary | None = None,
        feedback: str | None = None,
        instruction: str | None = None,
    ) -> DraftItinerary:
        """Draft an itinerary, or revise the previous one.

        Args:
            request: The trip being planned.
            brief: What the research agent established.
            previous: The draft under revision, when this is not the first pass.
            feedback: Reviewer comments driving a rejection.
            instruction: A free-text modification the reviewer asked for.

        Returns:
            A draft whose day numbering, dates and budget are assigned deterministically.
        """
        allocation = AllocationRecorder()
        tools = [
            build_budget_allocator_tool(self._context, request, allocation),
            build_day_feasibility_tool(
                self._settings,
                request.start_date,
                weather=brief.weather,
                allocation_source=allocation,
            ),
        ]

        agent: ToolCallingAgent[PlannerOutput] = ToolCallingAgent(
            name=_AGENT_NAME,
            registry=self._registry,
            tools=tools,
            output_schema=PlannerOutput,
            max_iterations=self._settings.agent_max_tool_iterations,
            extraction_attempts=self._settings.llm_extraction_attempts,
        )

        system_prompt = self._prompts.render(
            "planner_system",
            request=request,
            interests=", ".join(request.interests) or "not specified",
            brief=render_brief(brief),
            feedback=feedback,
            instruction=instruction,
            previous_plan=render_itinerary(previous) if previous else None,
        )
        task = (
            f"Build the {request.duration_days}-day itinerary for {request.destination}."
            if previous is None
            else f"Revise the {request.duration_days}-day itinerary for {request.destination}."
        )
        extraction_prompt = self._prompts.render(
            "planner_extract",
            duration_days=request.duration_days,
            start_date=request.start_date.isoformat(),
            currency=request.currency,
        )

        output = await agent.run(
            system_prompt, task, extraction_prompt, accept=lambda result: bool(result.activities)
        )
        itinerary = self._assemble(request, brief, output, allocation, previous)

        logger.info(
            "planner.draft_ready",
            destination=request.destination,
            days=itinerary.duration_days,
            revision=itinerary.revision,
            within_budget=itinerary.fits_budget(),
        )
        return itinerary

    def _assemble(
        self,
        request: TravelRequest,
        brief: ResearchBrief,
        output: PlannerOutput,
        allocation: AllocationRecorder,
        previous: DraftItinerary | None,
    ) -> DraftItinerary:
        """Build the domain itinerary by grouping the model's flat rows into days.

        Rows naming a day outside the trip are dropped rather than allowed to stretch
        the itinerary past its dates; that is a counting slip, not a planning decision.
        """
        outlook_days = brief.weather.days if brief.weather else []
        weather_by_date = {entry.day: entry for entry in outlook_days}
        duration = request.duration_days

        activities_by_day: dict[int, list[Activity]] = defaultdict(list)
        for row in output.activities:
            if 1 <= row.day <= duration:
                activities_by_day[row.day].append(row.to_activity())

        dining_by_day: dict[int, list[DiningSuggestion]] = defaultdict(list)
        for suggestion in output.dining:
            if 1 <= suggestion.day <= duration:
                dining_by_day[suggestion.day].append(suggestion.to_suggestion())

        dropped = sum(
            1 for row in output.activities if not 1 <= row.day <= duration
        ) + sum(1 for row in output.dining if not 1 <= row.day <= duration)
        if dropped:
            logger.warning("planner.rows_outside_trip_dropped", count=dropped, days=duration)

        days = [
            DayPlan(
                day_number=position,
                day_date=request.start_date + timedelta(days=position - 1),
                theme=(
                    output.day_themes[position - 1]
                    if position <= len(output.day_themes)
                    else ""
                ),
                activities=activities_by_day.get(position, []),
                dining=dining_by_day.get(position, []),
                notes=(
                    ""
                    if activities_by_day.get(position)
                    else "Left open during planning. Ask for a revision to fill it."
                ),
                weather=weather_by_date.get(request.start_date + timedelta(days=position - 1)),
            )
            for position in range(1, duration + 1)
        ]

        return DraftItinerary(
            destination=request.destination,
            start_date=request.start_date,
            end_date=request.end_date,
            travelers=request.travelers,
            summary=output.summary,
            days=days,
            budget=allocation.allocation,
            lodging_preference=output.lodging_preference,
            highlights=output.highlights,
            revision=previous.revision + 1 if previous else 0,
        )
