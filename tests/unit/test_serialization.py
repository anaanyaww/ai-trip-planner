"""Tests for checkpoint serialisation of the domain models.

Every domain object ends up inside a LangGraph checkpoint, so a field type that
validates correctly but cannot be encoded fails the entire workflow - and does so only
once real data arrives. That is what happened with ``pydantic.HttpUrl``: an empty brief
checkpointed fine and a brief carrying one real search result did not, so the offline
suite passed while every live run failed at the pause.

These tests therefore populate the fields that a live run populates and an offline run
leaves empty.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from itinera.domain.itinerary import (
    Activity,
    DayPlan,
    DiningSuggestion,
    DraftItinerary,
    LocalEvent,
)
from itinera.domain.research import (
    GeoLocation,
    PointOfInterest,
    ResearchBrief,
    ResearchFinding,
    SearchResult,
    WeatherOutlook,
    WeatherSource,
)
from itinera.persistence.serialization import build_serializer

URL = "https://example.com/kyoto"


@pytest.fixture
def serde():
    return build_serializer()


def _populated_brief() -> ResearchBrief:
    """Return a brief with every URL-carrying field filled, as a live run produces."""
    return ResearchBrief(
        destination="Kyoto, Japan",
        location=GeoLocation(name="Kyoto", country="Japan", latitude=35.01, longitude=135.77),
        attractions=[PointOfInterest(name="Nijo Castle", source_urls=[URL])],
        local_tips=[
            ResearchFinding(topic="transport", content="Buy an IC card.", source_urls=[URL])
        ],
        safety_notes=[ResearchFinding(topic="scams", content="Watch for fake monks.")],
        seasonal_notes=[ResearchFinding(topic="foliage", content="Peaks late November.")],
        weather=WeatherOutlook(source=WeatherSource.FORECAST),
        sources=[SearchResult(title="Kyoto guide", url=URL, provider="serper")],
    )


def test_a_brief_carrying_sources_round_trips(serde) -> None:
    """The case that broke live runs: an empty brief encoded, a populated one did not."""
    brief = _populated_brief()

    kind, blob = serde.dumps_typed(brief)
    restored = serde.loads_typed((kind, blob))

    assert isinstance(restored, ResearchBrief)
    assert restored.sources[0].url == URL
    assert restored.attractions[0].source_urls == [URL]
    assert restored.local_tips[0].source_urls == [URL]


def test_an_itinerary_carrying_source_urls_round_trips(serde) -> None:
    itinerary = DraftItinerary(
        destination="Kyoto, Japan",
        start_date=date(2026, 11, 10),
        end_date=date(2026, 11, 10),
        travelers=2,
        days=[
            DayPlan(
                day_number=1,
                day_date=date(2026, 11, 10),
                activities=[Activity(name="Nijo Castle", source_urls=[URL])],
                dining=[DiningSuggestion(name="Nishiki", source_url=URL)],
                events=[LocalEvent(name="Illumination", source_url=URL)],
            )
        ],
    )

    restored = serde.loads_typed(serde.dumps_typed(itinerary))

    assert restored.days[0].activities[0].source_urls == [URL]
    assert restored.days[0].dining[0].source_url == URL
    assert restored.days[0].events[0].source_url == URL


def test_urls_are_still_validated() -> None:
    """Serialisability must not have been bought by dropping validation."""
    with pytest.raises(ValidationError):
        SearchResult(title="bad", url="not-a-url")

    with pytest.raises(ValidationError):
        SearchResult(title="empty", url="")


def test_urls_are_normalised_the_way_pydantic_would() -> None:
    assert SearchResult(title="t", url="https://example.com").url == "https://example.com/"


def test_a_url_field_is_a_plain_string() -> None:
    """What made HttpUrl unserialisable was storing an object rather than a string."""
    result = SearchResult(title="t", url=URL)

    assert type(result.url) is str
    assert result.as_citation() == f"t ({URL})"


def test_every_checkpointed_domain_type_is_serialisable(serde) -> None:
    """A guard for the next field type added to the domain."""
    from itinera.persistence.serialization import _checkpointed_types

    unserialisable = []
    for model in _checkpointed_types():
        if not hasattr(model, "model_fields"):
            continue
        try:
            serde.dumps_typed(model.model_construct())
        except TypeError as exc:
            unserialisable.append(f"{model.__name__}: {exc}")

    assert not unserialisable


def test_a_model_supplied_non_url_is_dropped_not_fatal() -> None:
    """A bad citation must not cost the activity it was attached to."""
    from itinera.agents.schemas import PlannedActivity

    row = PlannedActivity(
        day=1,
        name="Kiyomizu-dera",
        source_urls=["n/a", "https://example.com/a", "see the guidebook"],
    )

    assert row.source_urls == ["https://example.com/a"]
    assert row.to_activity().name == "Kiyomizu-dera"


def test_a_degenerate_lodging_preference_is_rejected() -> None:
    """A model that loops on a phrase must not put kilobytes into the plan."""
    from datetime import date as _date

    from pydantic import ValidationError as _ValidationError

    with pytest.raises(_ValidationError):
        DraftItinerary(
            destination="Kyoto",
            start_date=_date(2026, 11, 10),
            end_date=_date(2026, 11, 10),
            travelers=1,
            lodging_preference="hotel choice stay preference " * 200,
        )
