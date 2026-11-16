"""Domain models shared across agents, graph and API.

Re-exported here so callers import from ``itinera.domain`` rather than reaching into
individual modules, which keeps the internal layout free to change.
"""

from itinera.domain.enums import (
    ActivityKind,
    BudgetCategory,
    PatchOperation,
    PlanStatus,
    ReviewActionType,
    RoutingTarget,
)
from itinera.domain.itinerary import (
    Activity,
    BudgetAllocation,
    BudgetLine,
    DayPlan,
    DiningSuggestion,
    DraftItinerary,
    FinalPlan,
    LocalEvent,
)
from itinera.domain.patching import apply_patches
from itinera.domain.research import (
    DailyWeather,
    GeoLocation,
    PointOfInterest,
    ResearchBrief,
    ResearchFinding,
    SearchResult,
    WeatherOutlook,
    WeatherSource,
)
from itinera.domain.review import ModificationPatch, ReviewDecision
from itinera.domain.travel import TravelRequest

__all__ = [
    "Activity",
    "ActivityKind",
    "BudgetAllocation",
    "BudgetCategory",
    "BudgetLine",
    "DailyWeather",
    "DayPlan",
    "DiningSuggestion",
    "DraftItinerary",
    "FinalPlan",
    "GeoLocation",
    "LocalEvent",
    "ModificationPatch",
    "PatchOperation",
    "PlanStatus",
    "PointOfInterest",
    "ResearchBrief",
    "ResearchFinding",
    "ReviewActionType",
    "ReviewDecision",
    "RoutingTarget",
    "SearchResult",
    "TravelRequest",
    "WeatherOutlook",
    "WeatherSource",
    "apply_patches",
]
