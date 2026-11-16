"""The workflow's nodes.

Each node is a small async function over the state, and the interesting one is
``await_review``. Calling :func:`langgraph.types.interrupt` there stops the graph mid
execution: LangGraph checkpoints everything, raises out of the run, and the payload
passed to ``interrupt`` becomes what the API surfaces to the reviewer. When a decision
arrives, the graph is resumed with a ``Command`` and this same node returns that
decision as its value. The pause is therefore durable rather than a held connection -
the process can restart between the two halves.

Nodes carry their dependencies in a :class:`WorkflowDependencies` container passed at
build time rather than importing them, so a test can drive the whole graph against the
offline provider without patching module state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from itinera.agents.planner.agent import PlannerAgent
from itinera.agents.rendering import render_itinerary, render_weather
from itinera.agents.research.agent import ResearchAgent
from itinera.agents.schemas import RoutingDecision, TripPreparation
from itinera.core.config import Settings
from itinera.core.exceptions import ItineraError, PlanValidationError
from itinera.core.logging import get_logger
from itinera.domain.enums import PlanStatus, ReviewActionType, RoutingTarget
from itinera.domain.itinerary import DraftItinerary, FinalPlan
from itinera.domain.patching import apply_patches
from itinera.domain.research import ResearchBrief
from itinera.domain.review import ReviewDecision
from itinera.domain.travel import TravelRequest
from itinera.graph.interrupts import runnable_context
from itinera.graph.state import PlanState, load, store
from itinera.llm.registry import ModelRegistry
from itinera.prompts.library import PromptLibrary
from itinera.services.progress import note
from itinera.tools.base import ToolContext

logger = get_logger(__name__)


@dataclass(frozen=True)
class WorkflowDependencies:
    """Everything the nodes need, injected once when the graph is built."""

    settings: Settings
    registry: ModelRegistry
    prompts: PromptLibrary
    tools: ToolContext

    @property
    def research_agent(self) -> ResearchAgent:
        """The destination research agent."""
        return ResearchAgent(self.settings, self.registry, self.prompts, self.tools)

    @property
    def planner_agent(self) -> PlannerAgent:
        """The itinerary planning agent."""
        return PlannerAgent(self.settings, self.registry, self.prompts, self.tools)


def _event(state: PlanState, event: str, **detail: Any) -> list[dict[str, Any]]:
    """Return the history with one entry appended.

    History is rebuilt rather than mutated because LangGraph merges a node's return
    into the checkpointed state; mutating the list in place would not be recorded.
    """
    entry = {
        "event": event,
        "at": datetime.now(timezone.utc).isoformat(),
        **detail,
    }
    return [*state.get("history", []), entry]


class WorkflowNodes:
    """The node implementations, bound to their dependencies."""

    def __init__(self, dependencies: WorkflowDependencies) -> None:
        """Store the injected dependencies."""
        self._deps = dependencies

    @staticmethod
    def _request(state: PlanState) -> TravelRequest:
        """Rebuild the travel request from its stored JSON."""
        return TravelRequest.model_validate(state["request"])

    async def validate_request(self, state: PlanState) -> PlanState:
        """Check the request against configured operational ceilings."""
        request = self._request(state)
        request.validate_against_limits(self._deps.settings)

        logger.info("workflow.validated", plan_id=state["plan_id"], **request.summary())
        return PlanState(
            status=PlanStatus.RESEARCHING,
            history=_event(state, "request_accepted", destination=request.destination),
        )

    async def research(self, state: PlanState) -> PlanState:
        """Gather destination intelligence into a brief."""
        note("research", "Researching the destination")
        brief = await self._deps.research_agent.run(
            self._request(state), feedback=state.get("feedback")
        )
        return PlanState(
            status=PlanStatus.PLANNING,
            brief=store(brief),
            history=_event(
                state,
                "research_complete",
                attractions=len(brief.attractions),
                sources=len(brief.sources),
            ),
        )

    async def plan(self, state: PlanState) -> PlanState:
        """Draft or revise the itinerary."""
        note("planner", "Drafting the itinerary")
        brief = load(ResearchBrief, state.get("brief"))
        if brief is None:  # pragma: no cover - the graph always researches first
            raise PlanValidationError("Planning was reached without a research brief")

        draft = await self._deps.planner_agent.run(
            self._request(state),
            brief,
            previous=load(DraftItinerary, state.get("draft")),
            feedback=state.get("feedback"),
            instruction=state.get("instruction"),
        )
        return PlanState(
            status=PlanStatus.AWAITING_REVIEW,
            draft=store(draft),
            feedback=None,
            instruction=None,
            history=_event(state, "draft_ready", revision=draft.revision),
        )

    async def await_review(self, state: PlanState, config: RunnableConfig) -> PlanState:
        """Pause for the human decision.

        The value handed to ``interrupt`` is what the API returns to the reviewer; the
        value it returns is the decision submitted to resume the workflow.

        Args:
            state: The workflow state carrying the draft under review.
            config: Passed by LangGraph; ``interrupt`` needs it in scope.
        """
        draft = load(DraftItinerary, state.get("draft"))
        if draft is None:  # pragma: no cover - the graph always plans first
            raise PlanValidationError("Review was reached without a draft")

        with runnable_context(config):
            payload = interrupt(
                {
                    "plan_id": state["plan_id"],
                    "revision": draft.revision,
                    "revisions_remaining": self._revisions_remaining(state),
                    "itinerary": draft.model_dump(mode="json"),
                    "rendered": render_itinerary(draft),
                }
            )
        decision = payload if isinstance(payload, ReviewDecision) else ReviewDecision(**payload)

        logger.info(
            "workflow.review_received",
            plan_id=state["plan_id"],
            action=decision.action.value,
        )
        return PlanState(
            decision=store(decision),
            feedback=decision.feedback,
            instruction=decision.instruction,
            history=_event(state, "review_submitted", **decision.audit_payload()),
        )

    async def apply_modification(self, state: PlanState) -> PlanState:
        """Apply structured patches without calling a model.

        A targeted edit costs no quota and produces a predictable result, so the
        reviewer sees exactly the change they asked for and nothing else.
        """
        draft = load(DraftItinerary, state.get("draft"))
        decision = load(ReviewDecision, state.get("decision"))
        assert draft is not None and decision is not None  # noqa: S101 - routing guarantees both

        try:
            patched = apply_patches(draft, decision.patches)
        except PlanValidationError as exc:
            logger.warning("workflow.patch_rejected", plan_id=state["plan_id"], error=str(exc))
            return PlanState(
                status=PlanStatus.AWAITING_REVIEW,
                error=exc.detail,
                history=_event(state, "modification_rejected", reason=exc.detail),
            )

        return PlanState(
            status=PlanStatus.AWAITING_REVIEW,
            draft=store(patched),
            error=None,
            revision_count=state.get("revision_count", 0) + 1,
            history=_event(
                state, "modification_applied", patches=len(decision.patches)
            ),
        )

    async def route_rejection(self, state: PlanState) -> PlanState:
        """Decide which agent should handle a rejection.

        The assignment asks for routing to "the appropriate agent(s)", so feedback is
        classified rather than always returned to the planner: a complaint about
        missing facts needs fresh research, whereas one about pacing does not. A
        failure to classify falls back to the planner, which is the cheaper path.
        """
        note("review", "Deciding which agent should handle the feedback")
        feedback = state.get("feedback") or ""
        target = RoutingTarget.PLANNER

        try:
            chain = self._deps.registry.structured_chain(RoutingDecision)
            decision = await chain.ainvoke(
                self._deps.prompts.render("feedback_router", feedback=feedback)
            )
            if isinstance(decision, RoutingDecision):
                target = RoutingTarget(decision.target.strip().lower())
        except (ValueError, ItineraError) as exc:
            logger.warning("workflow.routing_failed", error=str(exc))
        except Exception:
            logger.exception("workflow.routing_failed")

        logger.info("workflow.rejection_routed", plan_id=state["plan_id"], target=target.value)
        return PlanState(
            status=PlanStatus.RESEARCHING
            if target is RoutingTarget.RESEARCH
            else PlanStatus.REVISING,
            routing=target.value,
            revision_count=state.get("revision_count", 0) + 1,
            history=_event(state, "rejection_routed", target=target.value),
        )

    async def finalize(self, state: PlanState) -> PlanState:
        """Assemble the final plan from the approved itinerary."""
        note("finalize", "Assembling packing and preparation material")
        draft = load(DraftItinerary, state.get("draft"))
        assert draft is not None  # noqa: S101 - routing guarantees a draft exists
        brief = load(ResearchBrief, state.get("brief"))
        weather = render_weather(brief.weather if brief else None)

        preparation = await self._prepare(draft, weather)
        final = FinalPlan(
            itinerary=draft,
            packing_list=preparation.packing_list,
            preparation_checklist=preparation.preparation_checklist,
            weather_summary=weather,
            sources=brief.source_urls if brief else [],
            revision_count=state.get("revision_count", 0),
            revision_limit_reached=state.get("revision_limit_reached", False),
        )

        logger.info(
            "workflow.finalized",
            plan_id=state["plan_id"],
            revisions=final.revision_count,
            total_cost=final.total_estimated_cost,
        )
        return PlanState(
            status=PlanStatus.COMPLETED,
            final=store(final),
            history=_event(state, "plan_finalized", revisions=final.revision_count),
        )

    async def _prepare(self, draft: DraftItinerary, weather: str) -> TripPreparation:
        """Generate the packing and preparation lists.

        A failure here degrades to empty lists: the traveller has an approved itinerary,
        and losing the packing list is not a reason to lose the plan.
        """
        try:
            chain = self._deps.registry.structured_chain(TripPreparation)
            result = await chain.ainvoke(
                self._deps.prompts.render(
                    "finalizer", itinerary=render_itinerary(draft), weather=weather
                )
            )
        except Exception:
            logger.exception("workflow.preparation_failed")
            return TripPreparation()

        return result if isinstance(result, TripPreparation) else TripPreparation()

    def _revisions_remaining(self, state: PlanState) -> int:
        """Return how many further revision cycles the reviewer may request."""
        used = state.get("revision_count", 0)
        return max(self._deps.settings.max_revisions - used, 0)


    def route_after_review(self, state: PlanState) -> str:
        """Choose the next node from the reviewer's decision.

        Approval always finalises. Anything else consumes a revision, so once the
        configured ceiling is reached the workflow finalises the current draft rather
        than looping - and records that it was forced, so the response can say so.
        """
        decision = load(ReviewDecision, state.get("decision"))
        if decision is None or decision.action is ReviewActionType.APPROVE:
            return "finalize"

        if self._revisions_remaining(state) <= 0:
            logger.warning(
                "workflow.revision_limit_reached",
                plan_id=state["plan_id"],
                limit=self._deps.settings.max_revisions,
            )
            return "limit_reached"

        if decision.is_structured_modification:
            return "apply_modification"
        if decision.action is ReviewActionType.MODIFY:
            return "plan"
        return "route_rejection"

    @staticmethod
    def route_after_rejection(state: PlanState) -> str:
        """Send a routed rejection to the agent that was chosen for it."""
        return "research" if state.get("routing") == RoutingTarget.RESEARCH.value else "plan"

    async def mark_limit_reached(self, state: PlanState) -> PlanState:
        """Record that the revision ceiling forced the plan to finalise."""
        return PlanState(
            status=PlanStatus.FINALIZING,
            revision_limit_reached=True,
            history=_event(
                state,
                "revision_limit_reached",
                limit=self._deps.settings.max_revisions,
            ),
        )
