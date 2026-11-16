"""Assembly of the planning workflow.

The shape of the graph is the answer to the assignment's central question - how the
human-in-the-loop pause interacts with the API. Everything before ``await_review``
runs on the submission request; the pause checkpoints and unwinds; everything after it
runs on the review request. The two halves share nothing but the checkpointed state,
which is what allows the process to restart between them.

Rejections do not simply return to the planner. ``route_rejection`` classifies the
feedback first, so a complaint about missing facts re-runs research while a complaint
about pacing only re-runs planning. The revision ceiling is enforced on the same edge,
so a reviewer who keeps rejecting still ends up with a plan rather than a loop.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from itinera.core.logging import get_logger
from itinera.graph.nodes import WorkflowDependencies, WorkflowNodes
from itinera.graph.state import PlanState

PlanningGraph = CompiledStateGraph[PlanState, Any, Any, Any]

logger = get_logger(__name__)

NODE_VALIDATE = "validate"
NODE_RESEARCH = "research"
NODE_PLAN = "plan"
NODE_REVIEW = "await_review"
NODE_MODIFY = "apply_modification"
NODE_ROUTE_REJECTION = "route_rejection"
NODE_LIMIT_REACHED = "limit_reached"
NODE_FINALIZE = "finalize"


def build_graph(
    dependencies: WorkflowDependencies, checkpointer: BaseCheckpointSaver[str]
) -> PlanningGraph:
    """Compile the planning workflow.

    Args:
        dependencies: Settings, models, prompts and tool context for the nodes.
        checkpointer: Persistence for the paused state; without one, the interrupt
            could not be resumed on a later request.

    Returns:
        The compiled graph, ready to be invoked with a thread id per plan.
    """
    nodes = WorkflowNodes(dependencies)
    graph: StateGraph[PlanState, Any, Any, Any] = StateGraph(PlanState)

    graph.add_node(NODE_VALIDATE, nodes.validate_request)
    graph.add_node(NODE_RESEARCH, nodes.research)
    graph.add_node(NODE_PLAN, nodes.plan)
    graph.add_node(NODE_REVIEW, nodes.await_review)
    graph.add_node(NODE_MODIFY, nodes.apply_modification)
    graph.add_node(NODE_ROUTE_REJECTION, nodes.route_rejection)
    graph.add_node(NODE_LIMIT_REACHED, nodes.mark_limit_reached)
    graph.add_node(NODE_FINALIZE, nodes.finalize)

    graph.add_edge(START, NODE_VALIDATE)
    graph.add_edge(NODE_VALIDATE, NODE_RESEARCH)
    graph.add_edge(NODE_RESEARCH, NODE_PLAN)
    graph.add_edge(NODE_PLAN, NODE_REVIEW)

    graph.add_conditional_edges(
        NODE_REVIEW,
        nodes.route_after_review,
        {
            NODE_FINALIZE: NODE_FINALIZE,
            NODE_MODIFY: NODE_MODIFY,
            NODE_PLAN: NODE_PLAN,
            NODE_ROUTE_REJECTION: NODE_ROUTE_REJECTION,
            NODE_LIMIT_REACHED: NODE_LIMIT_REACHED,
        },
    )
    graph.add_conditional_edges(
        NODE_ROUTE_REJECTION,
        nodes.route_after_rejection,
        {NODE_RESEARCH: NODE_RESEARCH, NODE_PLAN: NODE_PLAN},
    )
    graph.add_edge(NODE_MODIFY, NODE_REVIEW)
    graph.add_edge(NODE_LIMIT_REACHED, NODE_FINALIZE)
    graph.add_edge(NODE_FINALIZE, END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("workflow.compiled", nodes=len(graph.nodes))
    return compiled
