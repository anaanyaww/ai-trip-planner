"""The workflow state.

Everything the orchestrator needs to resume after the human-in-the-loop pause lives
here, because this is exactly what the checkpointer persists. The pause can outlast the
process: the draft is written to SQLite when the graph interrupts, and the review that
resumes it arrives on a different HTTP request, possibly after a restart. Any context
held in a local variable instead of in this dict would not survive that.

**The state holds data, not objects.** Domain models are stored as the JSON their
``model_dump(mode="json")`` produces, and rebuilt by the node that needs them. This is
the single most important property of this module, and it was learned the hard way: the
checkpointer encodes with msgpack, which understands JSON primitives and nothing else.
Storing live models pushed pydantic objects - a URL, an enum, whatever a future field
introduces - into the encoder, where they failed. Worse, msgpack names only the
outermost type in its error, so every such failure reported the same useless
``DraftItinerary`` regardless of which field was actually at fault.

Round-tripping through JSON also means every read is re-validated, so a checkpoint
written by an older build cannot smuggle a stale shape into a running one.

The state is a ``TypedDict`` rather than a Pydantic model because LangGraph merges node
returns into it key by key; a node returns only the keys it changed, and the rest are
carried forward untouched.
"""

from __future__ import annotations

from typing import Any, TypedDict, TypeVar

from pydantic import BaseModel

from itinera.domain.enums import PlanStatus
from itinera.domain.travel import TravelRequest

ModelT = TypeVar("ModelT", bound=BaseModel)

Stored = dict[str, Any]


def store(model: BaseModel | None) -> Stored | None:
    """Return the JSON form of ``model`` for the checkpoint, or ``None``."""
    return None if model is None else model.model_dump(mode="json")


def load(schema: type[ModelT], data: Stored | None) -> ModelT | None:
    """Rebuild a model from its stored JSON, or return ``None`` when absent.

    Args:
        schema: The model to rebuild.
        data: The stored JSON, or ``None``.

    Returns:
        A validated instance, or ``None``.

    Raises:
        ValidationError: If the stored data no longer satisfies the model. That is a
            genuine incompatibility worth surfacing rather than papering over.
    """
    return None if data is None else schema.model_validate(data)


class PlanState(TypedDict, total=False):
    """State threaded through the planning workflow.

    Every model-shaped value is stored as JSON; use :func:`load` to rebuild it.

    Attributes:
        plan_id: Identifier shared by the API resource and the checkpoint thread.
        request: The validated travel request; the workflow's only input.
        status: Current stage, reported verbatim by ``GET /plan/{id}``.
        brief: What the research agent established.
        draft: The itinerary currently under review.
        final: The approved plan with its preparation material.
        decision: The review decision that resumed the last pause.
        routing: Which agent a rejection was routed to, as its enum value.
        revision_count: Completed review cycles, bounded by ``MAX_REVISIONS``.
        revision_limit_reached: Whether the last cycle was forced to finalise.
        feedback: Reviewer comments driving the current revision.
        instruction: A free-text modification the reviewer asked for.
        history: Append-only audit trail of decisions and transitions.
        error: Message explaining a failed run.
    """

    plan_id: str
    request: Stored
    status: PlanStatus
    brief: Stored | None
    draft: Stored | None
    final: Stored | None
    decision: Stored | None
    routing: str | None
    revision_count: int
    revision_limit_reached: bool
    feedback: str | None
    instruction: str | None
    history: list[dict[str, Any]]
    error: str | None


def initial_state(plan_id: str, request: TravelRequest) -> PlanState:
    """Build the state a new plan starts from.

    Every key is populated up front, so a node reading a value the workflow has not
    reached yet gets ``None`` rather than a ``KeyError``.
    """
    return PlanState(
        plan_id=plan_id,
        request=request.model_dump(mode="json"),
        status=PlanStatus.PENDING,
        brief=None,
        draft=None,
        final=None,
        decision=None,
        routing=None,
        revision_count=0,
        revision_limit_reached=False,
        feedback=None,
        instruction=None,
        history=[],
        error=None,
    )
