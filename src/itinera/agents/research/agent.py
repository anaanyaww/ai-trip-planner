"""The research agent.

Its output is the contract the planner works from, so the important design decision
here is what the model is *not* allowed to contribute. The resolved location, the
weather outlook and the list of sources all come from the recorders the tools wrote
to, never from the model's summary of what it saw. A model asked to repeat a URL back
will eventually repeat it wrongly, and a plan citing a URL that was never returned is
worse than one citing nothing.
"""

from __future__ import annotations

from itinera.agents.base import ToolCallingAgent
from itinera.agents.schemas import ResearchOutput
from itinera.core.config import Settings
from itinera.core.logging import get_logger
from itinera.domain.research import ResearchBrief
from itinera.domain.travel import TravelRequest
from itinera.llm.registry import ModelRegistry
from itinera.prompts.library import PromptLibrary
from itinera.tools.base import ToolContext
from itinera.tools.destination_conditions import (
    ConditionsRecorder,
    build_destination_conditions_tool,
)
from itinera.tools.web_search import SourceRecorder, build_web_search_tool

logger = get_logger(__name__)

_AGENT_NAME = "research"


class ResearchAgent:
    """Gathers destination intelligence into a typed brief."""

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

    async def run(self, request: TravelRequest, feedback: str | None = None) -> ResearchBrief:
        """Research ``request`` and return a grounded brief.

        Fresh recorders are created per run, so a revision cannot inherit sources from
        the run that preceded it.

        Args:
            request: The trip being researched.
            feedback: Reviewer comments when this run is a revision.

        Returns:
            A brief whose factual anchors come from the tools rather than the model.
        """
        sources = SourceRecorder()
        conditions = ConditionsRecorder()
        tools = [
            build_web_search_tool(self._context, sources),
            build_destination_conditions_tool(self._context, conditions),
        ]

        agent: ToolCallingAgent[ResearchOutput] = ToolCallingAgent(
            name=_AGENT_NAME,
            registry=self._registry,
            tools=tools,
            output_schema=ResearchOutput,
            max_iterations=self._settings.agent_max_tool_iterations,
            extraction_attempts=self._settings.llm_extraction_attempts,
        )

        system_prompt = self._prompts.render(
            "research_system",
            request=request,
            interests=", ".join(request.interests) or "not specified",
            search_budget=self._settings.search_max_queries_per_run,
            feedback=feedback,
        )
        task = (
            f"Research {request.destination} for a {request.duration_days}-day trip "
            f"starting {request.start_date.isoformat()}."
        )
        output = await agent.run(
            system_prompt,
            task,
            self._prompts.render("research_extract"),
            accept=lambda result: bool(result.attractions or result.local_tips),
        )

        brief = ResearchBrief(
            destination=request.destination,
            location=conditions.location,
            attractions=output.attractions,
            local_tips=output.local_tips,
            safety_notes=output.safety_notes,
            seasonal_notes=output.seasonal_notes,
            weather=conditions.outlook,
            sources=sources.results,
        )
        logger.info(
            "research.brief_ready",
            destination=request.destination,
            attractions=len(brief.attractions),
            sources=len(brief.sources),
            usable=brief.is_usable,
        )
        return brief
