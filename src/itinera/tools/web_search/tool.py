"""The web search tool exposed to the research agent.

Two concerns beyond simply calling the provider are handled here.

The first is quota. Serper's free tier is a lifetime allowance, not a monthly one, and
an agent left to its own judgement will happily issue a dozen searches for a
three-day trip. A per-run budget caps that, and refusing further searches with an
explanatory message rather than an error lets the agent wind down gracefully and work
with what it already has.

The second is traceability. Rather than trusting the model to copy URLs accurately out
of tool output, every URL the provider actually returned is recorded out-of-band. The
final plan cites that record, so a source in the plan is a page that genuinely appeared
in a search result.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from itinera.core.exceptions import ItineraError
from itinera.core.logging import get_logger
from itinera.domain.research import SearchResult
from itinera.tools.base import ToolContext, guarded_call
from itinera.tools.web_search.provider import build_search_provider

logger = get_logger(__name__)

_TOOL_NAME = "web_search"


class SourceRecorder:
    """Collects the results a run's searches actually returned.

    One instance is created per workflow run and closed over by the tool, so the
    record cannot leak between plans.
    """

    def __init__(self) -> None:
        """Start with an empty, de-duplicated record."""
        self._results: dict[str, SearchResult] = {}

    def record(self, results: list[SearchResult]) -> None:
        """Add results, keeping the first sighting of each URL."""
        for result in results:
            self._results.setdefault(str(result.url), result)

    @property
    def results(self) -> list[SearchResult]:
        """Every distinct result seen this run, in first-seen order."""
        return list(self._results.values())

    def __len__(self) -> int:
        """Return how many distinct results have been recorded."""
        return len(self._results)


class WebSearchInput(BaseModel):
    """Arguments accepted by the web search tool."""

    query: str = Field(
        description=(
            "A focused search query. Prefer several specific queries over one broad "
            "one, for example 'Kyoto temple etiquette rules' rather than 'Kyoto info'."
        )
    )


def _format_results(results: list[SearchResult]) -> str:
    """Render results as a numbered list the model can cite from."""
    if not results:
        return "No results found. Try a different phrasing or a narrower query."

    lines: list[str] = []
    for position, result in enumerate(results, start=1):
        snippet = result.snippet.strip() or "(no snippet available)"
        lines.append(f"{position}. {result.title}\n   {snippet}\n   Source: {result.url}")
    return "\n".join(lines)


def build_web_search_tool(
    context: ToolContext, recorder: SourceRecorder | None = None
) -> BaseTool:
    """Build the web search tool bound to a run's context and source record.

    Args:
        context: Injected settings, HTTP client, cache and breakers.
        recorder: Collector for the URLs this run actually saw.

    Returns:
        A LangChain tool the research agent can call.
    """
    provider = build_search_provider(context.settings, context.http)
    budget = context.settings.search_max_queries_per_run
    limit = context.settings.search_results_per_query
    remaining = {"queries": budget}

    async def _search(query: str) -> str:
        """Search the web and return formatted results."""
        cleaned = query.strip()
        if not cleaned:
            return "The query was empty. Supply a specific question to search for."

        if remaining["queries"] <= 0:
            logger.info("search.budget_exhausted", budget=budget)
            return (
                f"The search budget of {budget} queries for this trip is exhausted. "
                "Draw your conclusions from the results already gathered."
            )

        try:
            payload = await guarded_call(
                context,
                service=provider.name,
                cache_namespace=f"{provider.name}:search",
                cache_params={"query": cleaned.lower(), "limit": limit},
                ttl_seconds=context.settings.cache_ttl_search_seconds,
                operation=lambda: provider.fetch(cleaned, limit),
            )
        except ItineraError as exc:
            logger.warning("search.failed", query=cleaned, error=str(exc))
            return f"The search failed ({exc}). Continue with the information you have."

        results = provider.parse(payload, limit)

        remaining["queries"] -= 1
        if recorder is not None:
            recorder.record(results)

        logger.info("search.completed", query=cleaned, results=len(results))
        return _format_results(results)

    return StructuredTool.from_function(
        coroutine=_search,
        name=_TOOL_NAME,
        description=(
            "Search the live web for current information about a destination: "
            "attractions, local customs, safety guidance, opening hours, costs and "
            "seasonal considerations. Returns titles, snippets and source URLs."
        ),
        args_schema=WebSearchInput,
    )
