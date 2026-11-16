"""Tests for the prompt library.

The failure worth guarding against is silent: a template that references context the
caller did not supply. Rendering it to an empty string would produce a prompt missing
the trip's dates or budget, and the only symptom would be a worse plan. StrictUndefined
turns that into an exception, and this asserts it stays that way.
"""

from __future__ import annotations

import pytest
from jinja2 import UndefinedError

from itinera.core.config import Settings
from itinera.core.exceptions import ConfigurationError
from itinera.domain.travel import TravelRequest
from itinera.prompts.library import PromptLibrary, get_prompt_library

REQUIRED_PROMPTS = {
    "research_system",
    "research_extract",
    "planner_system",
    "planner_extract",
    "feedback_router",
    "finalizer",
}


@pytest.fixture
def library(settings: Settings) -> PromptLibrary:
    return get_prompt_library(settings)


def test_every_prompt_the_workflow_uses_exists(library: PromptLibrary) -> None:
    assert set(library.names()) >= REQUIRED_PROMPTS


def test_an_unknown_prompt_version_fails_at_construction() -> None:
    """A typo in PROMPT_VERSION should be caught at start-up, not at the first run."""
    with pytest.raises(ConfigurationError, match="no template directory"):
        PromptLibrary("v-does-not-exist")


def test_an_unknown_prompt_names_the_version(library: PromptLibrary) -> None:
    with pytest.raises(ConfigurationError, match="does not exist in version"):
        library.render("no_such_prompt")


def test_missing_context_raises_rather_than_rendering_blank(library: PromptLibrary) -> None:
    with pytest.raises(UndefinedError):
        library.render("research_system")


def test_the_research_prompt_carries_the_trip_facts(
    library: PromptLibrary, travel_request: TravelRequest, settings: Settings
) -> None:
    rendered = library.render(
        "research_system",
        request=travel_request,
        interests="food, temples",
        search_budget=settings.search_max_queries_per_run,
        feedback=None,
    )

    assert travel_request.destination in rendered
    assert travel_request.start_date.isoformat() in rendered
    assert str(settings.search_max_queries_per_run) in rendered
    assert "Revision requested" not in rendered


def test_feedback_is_included_only_when_a_revision_is_under_way(
    library: PromptLibrary, travel_request: TravelRequest, settings: Settings
) -> None:
    rendered = library.render(
        "research_system",
        request=travel_request,
        interests="food",
        search_budget=settings.search_max_queries_per_run,
        feedback="Nothing covers the food markets.",
    )

    assert "Revision requested" in rendered
    assert "Nothing covers the food markets." in rendered


def test_the_planner_prompt_separates_feedback_from_a_modification(
    library: PromptLibrary, travel_request: TravelRequest
) -> None:
    """A rejection and a targeted edit ask for different things, so they read differently."""
    modification = library.render(
        "planner_system",
        request=travel_request,
        interests="food",
        brief="A brief.",
        feedback=None,
        instruction="Swap the day 4 museum for something outdoors.",
        previous_plan="Day 1: museum",
        duration_days=travel_request.duration_days,
    )

    assert "Modification requested" in modification
    assert "change nothing else" in modification
    assert "Revision requested" not in modification
    assert "Day 1: museum" in modification


def test_rendering_is_stripped_and_stable(
    library: PromptLibrary, travel_request: TravelRequest
) -> None:
    context = {
        "request": travel_request,
        "interests": "food",
        "brief": "A brief.",
        "feedback": None,
        "instruction": None,
        "previous_plan": None,
    }
    first = library.render("planner_system", **context)
    second = library.render("planner_system", **context)

    assert first == second
    assert first == first.strip()


def test_the_library_is_cached_per_version(settings: Settings) -> None:
    """Agents ask for prompts every turn; rebuilding the environment each time is waste."""
    assert get_prompt_library(settings) is get_prompt_library(settings)


def test_the_rendered_brief_is_bounded(travel_request: TravelRequest) -> None:
    """An unbounded brief turns a revision into a request free tiers reject outright."""
    from itinera.agents.rendering import render_brief
    from itinera.domain.research import PointOfInterest, ResearchBrief, ResearchFinding

    brief = ResearchBrief(
        destination="Kyoto",
        attractions=[PointOfInterest(name=f"Place {n}") for n in range(60)],
        local_tips=[
            ResearchFinding(topic=f"tip {n}", content="x" * 2000) for n in range(30)
        ],
    )

    rendered = render_brief(brief)

    assert rendered.count("Place ") == 20
    assert "further findings omitted" in rendered
    assert len(rendered) < 20_000, "the brief must not dominate the prompt budget"
