"""Output of the research agent.

The research brief is the contract between the two agents: everything the planner
knows about the destination arrives through this structure. Keeping it typed rather
than passing prose between agents means the planner receives named fields it can
reason over, and means a malformed research step fails at the boundary instead of
producing a plausible-looking but ungrounded itinerary.

Every substantive finding carries its source URLs. That is what allows the final plan
to be traced back to real pages rather than to the model's recollection.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from itinera.domain.enums import ActivityKind
from itinera.domain.types import SourceUrl


class SearchResult(BaseModel):
    """A single web-search hit, normalised across search providers."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: SourceUrl
    snippet: str = ""
    provider: str = "unknown"

    def as_citation(self) -> str:
        """Render the result as a compact citation line for prompt context."""
        return f"{self.title} ({self.url})"


class GeoLocation(BaseModel):
    """A resolved geographic location.

    Coordinates come from geocoding the free-text destination, which is what lets the
    weather lookup and any distance reasoning work from a real point on the map rather
    than from the destination string.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    country: str | None = None
    admin_area: str | None = None
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    timezone: str | None = None

    @property
    def label(self) -> str:
        """Human-readable location label, narrowing to the country when known."""
        return f"{self.name}, {self.country}" if self.country else self.name


class WeatherSource(str, Enum):
    """Where a weather outlook came from.

    A genuine forecast is only available inside the provider's horizon; beyond it the
    system reports climate normals for the same calendar window and says so, rather
    than presenting a guess as a forecast.
    """

    FORECAST = "forecast"
    CLIMATE_NORMALS = "climate_normals"


class DailyWeather(BaseModel):
    """Weather expectation for one day of the trip."""

    model_config = ConfigDict(frozen=True)

    day: date
    temp_min_c: float
    temp_max_c: float
    precipitation_mm: Annotated[float, Field(ge=0)] = 0.0

    @property
    def is_wet(self) -> bool:
        """Whether precipitation is high enough to favour indoor activities."""
        return self.precipitation_mm >= 5.0


class WeatherOutlook(BaseModel):
    """Aggregated weather expectation across the travel window."""

    model_config = ConfigDict(frozen=True)

    source: WeatherSource
    days: list[DailyWeather] = Field(default_factory=list)
    summary: str = ""

    @property
    def average_high_c(self) -> float | None:
        """Mean daily maximum across the window, or ``None`` with no data."""
        if not self.days:
            return None
        return round(sum(day.temp_max_c for day in self.days) / len(self.days), 1)

    @property
    def average_low_c(self) -> float | None:
        """Mean daily minimum across the window, or ``None`` with no data."""
        if not self.days:
            return None
        return round(sum(day.temp_min_c for day in self.days) / len(self.days), 1)

    @property
    def wet_days(self) -> list[date]:
        """Dates wet enough that an indoor alternative should be planned."""
        return [day.day for day in self.days if day.is_wet]


class PointOfInterest(BaseModel):
    """A candidate activity the planner may schedule."""

    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(min_length=1, max_length=160)]
    kind: ActivityKind = ActivityKind.SIGHTSEEING
    description: str = ""
    neighbourhood: str | None = None
    typical_duration_hours: Annotated[float, Field(gt=0, le=24)] = 2.0
    estimated_cost_per_person: float | None = Field(default=None, ge=0)
    source_urls: list[SourceUrl] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        return value.strip()


class ResearchFinding(BaseModel):
    """A grounded observation about the destination on one research topic."""

    model_config = ConfigDict(frozen=True)

    topic: Annotated[str, Field(min_length=1, max_length=80)]
    content: Annotated[str, Field(min_length=1)]
    source_urls: list[SourceUrl] = Field(default_factory=list)


class ResearchBrief(BaseModel):
    """Everything the research agent learned about the destination.

    The topic fields mirror the coverage the assignment asks for - attractions, local
    tips, safety, weather and seasonal considerations - so a missing area is visible
    as an empty field rather than being silently absent from a wall of prose.
    """

    destination: str
    location: GeoLocation | None = None
    attractions: list[PointOfInterest] = Field(default_factory=list)
    local_tips: list[ResearchFinding] = Field(default_factory=list)
    safety_notes: list[ResearchFinding] = Field(default_factory=list)
    seasonal_notes: list[ResearchFinding] = Field(default_factory=list)
    weather: WeatherOutlook | None = None
    sources: list[SearchResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    revision: Annotated[int, Field(ge=0)] = 0

    @property
    def is_usable(self) -> bool:
        """Whether the brief carries enough substance for the planner to work from."""
        return bool(self.attractions) or bool(self.local_tips)

    @property
    def source_urls(self) -> list[str]:
        """Distinct source URLs cited anywhere in the brief, in first-seen order."""
        seen: dict[str, None] = {}
        for result in self.sources:
            seen.setdefault(str(result.url), None)
        for group in (self.local_tips, self.safety_notes, self.seasonal_notes):
            for finding in group:
                for url in finding.source_urls:
                    seen.setdefault(str(url), None)
        return list(seen)
