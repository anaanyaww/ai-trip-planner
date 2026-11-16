"""Destination conditions: the research agent's second tool.

Weather drives real planning decisions - whether a day should be outdoors, whether an
indoor alternative is needed, what to pack - and it is exactly the kind of fact a
language model will invent confidently. This tool grounds it.

Open-Meteo was chosen over the better-known weather APIs because it needs no API key
at all, so this capability works on a fresh clone with an empty ``.env``.

The distinction the tool is careful about: a genuine forecast only exists inside the
provider's horizon, roughly sixteen days. Beyond that it computes climate normals by
averaging the same calendar window across recent years, and labels the result as such.
Presenting a multi-month-out average as a forecast would be the same fabrication the
tool exists to prevent.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from itinera.core.exceptions import ItineraError, ToolExecutionError
from itinera.core.logging import get_logger
from itinera.domain.research import DailyWeather, GeoLocation, WeatherOutlook, WeatherSource
from itinera.tools.base import ToolContext, guarded_call

logger = get_logger(__name__)

_TOOL_NAME = "destination_conditions"
_DAILY_VARIABLES = "temperature_2m_max,temperature_2m_min,precipitation_sum"
_LEAP_DAY = (2, 29)


def _shift_year(day: date, years_back: int) -> date:
    """Return ``day`` moved back by ``years_back`` years.

    29 February has no counterpart in a common year, so it maps to the 28th.
    """
    try:
        return day.replace(year=day.year - years_back)
    except ValueError:
        return day.replace(year=day.year - years_back, month=_LEAP_DAY[0], day=28)


class ConditionsRecorder:
    """Holds the structured lookup a run performed, for use in the research brief.

    The agent receives prose; the workflow needs the underlying objects. Recording
    them here avoids re-parsing the model's rendering of numbers it was given.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.location: GeoLocation | None = None
        self.outlook: WeatherOutlook | None = None

    def record(self, location: GeoLocation | None, outlook: WeatherOutlook | None) -> None:
        """Store the most recent successful lookup."""
        if location is not None:
            self.location = location
        if outlook is not None:
            self.outlook = outlook


class DestinationConditionsInput(BaseModel):
    """Arguments accepted by the destination conditions tool."""

    destination: str = Field(description="City or place name, for example 'Kyoto, Japan'.")
    start_date: date = Field(description="First day of the trip, as YYYY-MM-DD.")
    end_date: date = Field(description="Last day of the trip, as YYYY-MM-DD.")


@dataclass(frozen=True)
class ConditionsResult:
    """A resolved location together with its weather outlook."""

    location: GeoLocation
    outlook: WeatherOutlook


class DestinationConditionsService:
    """Geocodes a destination and derives its weather outlook."""

    def __init__(self, context: ToolContext) -> None:
        """Store the injected dependencies."""
        self._context = context
        self._settings = context.settings

    async def geocode(self, destination: str) -> GeoLocation:
        """Resolve a free-text destination to coordinates.

        Args:
            destination: City or place name.

        Returns:
            The best-matching location.

        Raises:
            ToolExecutionError: If the destination cannot be resolved.
        """
        payload = await guarded_call(
            self._context,
            service="open-meteo-geocoding",
            cache_namespace="open-meteo:geocode",
            cache_params={"name": destination.strip().lower()},
            ttl_seconds=self._settings.cache_ttl_climate_seconds,
            operation=lambda: self._context.http.request_json(
                "GET",
                self._settings.open_meteo_geocoding_url,
                params={"name": destination.strip(), "count": 1, "format": "json"},
                service="open-meteo-geocoding",
            ),
        )

        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            raise ToolExecutionError(
                f"Could not resolve {destination!r} to a location. "
                "Try a more specific name, such as including the country.",
                destination=destination,
            )

        top = results[0]
        return GeoLocation(
            name=top.get("name", destination),
            country=top.get("country"),
            admin_area=top.get("admin1"),
            latitude=top["latitude"],
            longitude=top["longitude"],
            timezone=top.get("timezone"),
        )

    async def _fetch_daily(
        self, url: str, location: GeoLocation, start: date, end: date, service: str, ttl: int
    ) -> dict[str, Any]:
        """Fetch a daily weather series for a date range."""
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": _DAILY_VARIABLES,
            "timezone": "auto",
        }
        payload = await guarded_call(
            self._context,
            service=service,
            cache_namespace=f"open-meteo:{service}",
            cache_params=params,
            ttl_seconds=ttl,
            operation=lambda: self._context.http.request_json(
                "GET", url, params=params, service=service
            ),
        )
        daily = payload.get("daily") if isinstance(payload, dict) else None
        if not isinstance(daily, dict):
            raise ToolExecutionError(f"{service} returned no daily series", service=service)
        return daily

    @staticmethod
    def _rows(daily: dict[str, Any]) -> list[tuple[date, float, float, float]]:
        """Zip the parallel arrays Open-Meteo returns into per-day tuples.

        Days with a missing reading are dropped rather than defaulted, so a gap in the
        upstream series never becomes a fabricated zero.
        """
        times = daily.get("time", []) or []
        highs = daily.get("temperature_2m_max", []) or []
        lows = daily.get("temperature_2m_min", []) or []
        rain = daily.get("precipitation_sum", []) or []

        rows: list[tuple[date, float, float, float]] = []
        for index, stamp in enumerate(times):
            high = highs[index] if index < len(highs) else None
            low = lows[index] if index < len(lows) else None
            if high is None or low is None:
                continue
            precipitation = rain[index] if index < len(rain) else None
            rows.append(
                (date.fromisoformat(stamp), float(high), float(low), float(precipitation or 0.0))
            )
        return rows

    async def _forecast(self, location: GeoLocation, start: date, end: date) -> WeatherOutlook:
        """Return the genuine forecast for a near-term travel window."""
        daily = await self._fetch_daily(
            self._settings.open_meteo_forecast_url,
            location,
            start,
            end,
            service="forecast",
            ttl=self._settings.cache_ttl_forecast_seconds,
        )
        days = [
            DailyWeather(day=day, temp_max_c=high, temp_min_c=low, precipitation_mm=rain)
            for day, high, low, rain in self._rows(daily)
        ]
        return WeatherOutlook(
            source=WeatherSource.FORECAST,
            days=days,
            summary=f"Forecast for {location.label} covering {len(days)} day(s).",
        )

    async def _climate_normals(
        self, location: GeoLocation, start: date, end: date
    ) -> WeatherOutlook:
        """Average the same calendar window across recent years.

        Each historical year is fetched separately and aligned by position within the
        window, so the average describes "the third day of this trip" rather than a
        specific past date.
        """
        years = self._settings.climate_normal_years
        buckets: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

        async def fetch_year(years_back: int) -> dict[str, Any] | None:
            """Fetch one historical year, or ``None`` when it is unavailable."""
            historical_start = _shift_year(start, years_back)
            historical_end = _shift_year(end, years_back)
            try:
                return await self._fetch_daily(
                    self._settings.open_meteo_archive_url,
                    location,
                    historical_start,
                    historical_end,
                    service="archive",
                    ttl=self._settings.cache_ttl_climate_seconds,
                )
            except ItineraError as exc:
                logger.warning(
                    "climate.year_unavailable", year=historical_start.year, error=str(exc)
                )
                return None

        for daily in await asyncio.gather(*(fetch_year(back) for back in range(1, years + 1))):
            if daily is None:
                continue
            for offset, (_, high, low, rain) in enumerate(self._rows(daily)):
                buckets[offset].append((high, low, rain))

        if not buckets:
            raise ToolExecutionError(
                f"No historical weather is available for {location.label}",
                destination=location.label,
            )

        days: list[DailyWeather] = []
        for offset in sorted(buckets):
            samples = buckets[offset]
            count = len(samples)
            day = date.fromordinal(start.toordinal() + offset)
            days.append(
                DailyWeather(
                    day=day,
                    temp_max_c=round(sum(item[0] for item in samples) / count, 1),
                    temp_min_c=round(sum(item[1] for item in samples) / count, 1),
                    precipitation_mm=round(sum(item[2] for item in samples) / count, 1),
                )
            )

        return WeatherOutlook(
            source=WeatherSource.CLIMATE_NORMALS,
            days=days,
            summary=(
                f"Climate normals for {location.label}, averaged over the same calendar "
                f"window across the last {years} year(s). This is a seasonal expectation, "
                "not a forecast."
            ),
        )

    async def lookup(self, destination: str, start: date, end: date) -> ConditionsResult:
        """Resolve the destination and return the appropriate weather outlook.

        Args:
            destination: City or place name.
            start: First day of the trip.
            end: Last day of the trip.

        Returns:
            The resolved location and its outlook.

        Raises:
            ToolExecutionError: If the destination cannot be resolved or no weather
                data is available.
        """
        location = await self.geocode(destination)
        horizon = self._settings.forecast_horizon_days
        within_forecast_range = (end - date.today()).days <= horizon

        outlook = (
            await self._forecast(location, start, end)
            if within_forecast_range
            else await self._climate_normals(location, start, end)
        )
        logger.info(
            "conditions.resolved",
            destination=location.label,
            source=outlook.source.value,
            days=len(outlook.days),
        )
        return ConditionsResult(location=location, outlook=outlook)


def _format(result: ConditionsResult) -> str:
    """Render the outlook as prose the agent can reason over."""
    outlook = result.outlook
    lines = [
        f"Location: {result.location.label} "
        f"({result.location.latitude:.2f}, {result.location.longitude:.2f})",
        outlook.summary,
    ]
    if outlook.average_high_c is not None:
        lines.append(
            f"Average daily high {outlook.average_high_c} C, average low {outlook.average_low_c} C."
        )
    wet = outlook.wet_days
    lines.append(
        "Wet days likely: " + ", ".join(day.isoformat() for day in wet)
        if wet
        else "No notably wet days expected."
    )
    for day in outlook.days:
        lines.append(
            f"  {day.day.isoformat()}: {day.temp_min_c} to {day.temp_max_c} C, "
            f"{day.precipitation_mm} mm precipitation"
        )
    return "\n".join(lines)


def build_destination_conditions_tool(
    context: ToolContext, recorder: ConditionsRecorder | None = None
) -> BaseTool:
    """Build the destination conditions tool.

    Args:
        context: Injected settings, HTTP client, cache and breakers.
        recorder: Collector for the structured lookup, for use in the research brief.

    Returns:
        A LangChain tool the research agent can call.
    """
    service = DestinationConditionsService(context)

    async def _conditions(destination: str, start_date: date, end_date: date) -> str:
        """Look up location and weather for a travel window."""
        try:
            result = await service.lookup(destination, start_date, end_date)
        except ItineraError as exc:
            logger.warning("conditions.failed", destination=destination, error=str(exc))
            return (
                f"Could not retrieve conditions for {destination!r} ({exc}). "
                "Plan without weather-specific assumptions and say so."
            )

        if recorder is not None:
            recorder.record(result.location, result.outlook)
        return _format(result)

    return StructuredTool.from_function(
        coroutine=_conditions,
        name=_TOOL_NAME,
        description=(
            "Look up a destination's coordinates and the weather expected during the "
            "travel window. Returns a real forecast for near-term trips and clearly "
            "labelled seasonal climate normals for trips further out. Use this before "
            "deciding which activities should be outdoors."
        ),
        args_schema=DestinationConditionsInput,
    )
