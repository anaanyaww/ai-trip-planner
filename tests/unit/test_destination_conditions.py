"""Tests for the destination conditions tool.

The behaviour worth protecting is the forecast/climate-normals distinction: presenting
a multi-month-out average as a forecast is exactly the fabrication this tool exists to
prevent, so the routing and the labelling are both asserted.
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest

from itinera.core.exceptions import ToolExecutionError
from itinera.domain.research import WeatherSource
from itinera.tools.destination_conditions import (
    ConditionsRecorder,
    DestinationConditionsService,
    build_destination_conditions_tool,
)
from tests.conftest import Handler

GEOCODE_PAYLOAD = {
    "results": [
        {
            "name": "Kyoto",
            "country": "Japan",
            "admin1": "Kyoto",
            "latitude": 35.01,
            "longitude": 135.77,
            "timezone": "Asia/Tokyo",
        }
    ]
}


def _daily(days: list[date], highs: list[float], lows: list[float], rain: list[float]) -> dict:
    return {
        "daily": {
            "time": [day.isoformat() for day in days],
            "temperature_2m_max": highs,
            "temperature_2m_min": lows,
            "precipitation_sum": rain,
        }
    }


def _router(**routes: object) -> Handler:
    """Return a handler dispatching on the substring found in the request URL."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        for fragment, payload in routes.items():
            if fragment in str(request.url):
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": "unrouted"})

    handler.calls = calls
    return handler


async def test_near_term_window_returns_a_real_forecast(make_context) -> None:
    start = date.today() + timedelta(days=3)
    days = [start, start + timedelta(days=1)]
    handler = _router(
        geocoding=GEOCODE_PAYLOAD,
        forecast=_daily(days, [18.0, 20.0], [9.0, 11.0], [0.0, 7.0]),
    )
    service = DestinationConditionsService(make_context(handler))

    result = await service.lookup("Kyoto", start, days[-1])

    assert result.outlook.source is WeatherSource.FORECAST
    assert result.location.label == "Kyoto, Japan"
    assert [day.temp_max_c for day in result.outlook.days] == [18.0, 20.0]
    assert result.outlook.wet_days == [days[1]], "7mm should count as a wet day"


async def test_distant_window_falls_back_to_climate_normals(make_context, settings) -> None:
    start = date.today() + timedelta(days=200)
    end = start + timedelta(days=1)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocoding" in url:
            return httpx.Response(200, json=GEOCODE_PAYLOAD)
        if "archive" in url:
            historical = date.fromisoformat(request.url.params["start_date"])
            return httpx.Response(
                200,
                json=_daily(
                    [historical, historical + timedelta(days=1)],
                    [20.0, 24.0],
                    [10.0, 14.0],
                    [1.0, 3.0],
                ),
            )
        return httpx.Response(500, json={"error": "forecast must not be called"})

    service = DestinationConditionsService(make_context(handler))

    result = await service.lookup("Kyoto", start, end)

    outlook = result.outlook
    assert outlook.source is WeatherSource.CLIMATE_NORMALS
    assert "not a forecast" in outlook.summary
    assert [day.day for day in outlook.days] == [start, end], (
        "normals must be aligned to the trip's own dates, not the historical ones"
    )
    assert outlook.average_high_c == 22.0


async def test_climate_normals_survive_a_missing_year(make_context, settings) -> None:
    """One unavailable historical year should narrow the average, not fail the lookup."""
    start = date.today() + timedelta(days=200)
    seen_years: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocoding" in url:
            return httpx.Response(200, json=GEOCODE_PAYLOAD)
        historical = date.fromisoformat(request.url.params["start_date"])
        seen_years.append(historical.year)
        if historical.year == date.today().year - 1:
            return httpx.Response(503, json={"error": "archive unavailable"})
        return httpx.Response(200, json=_daily([historical], [20.0], [10.0], [0.0]))

    service = DestinationConditionsService(make_context(handler))

    outlook = (await service.lookup("Kyoto", start, start)).outlook

    assert outlook.source is WeatherSource.CLIMATE_NORMALS
    assert len(seen_years) == settings.climate_normal_years


async def test_unresolvable_destination_is_reported_clearly(make_context) -> None:
    service = DestinationConditionsService(make_context(_router(geocoding={"results": []})))

    with pytest.raises(ToolExecutionError, match="Could not resolve"):
        await service.geocode("Atlantis")


async def test_rows_with_missing_readings_are_dropped() -> None:
    rows = DestinationConditionsService._rows(
        {
            "time": ["2030-01-01", "2030-01-02", "2030-01-03"],
            "temperature_2m_max": [10.0, None, 12.0],
            "temperature_2m_min": [2.0, 1.0, None],
            "precipitation_sum": [0.0, 0.0, 0.0],
        }
    )

    assert [row[0] for row in rows] == [date(2030, 1, 1)]


async def test_tool_records_the_structured_lookup(make_context) -> None:
    start = date.today() + timedelta(days=3)
    handler = _router(
        geocoding=GEOCODE_PAYLOAD, forecast=_daily([start], [18.0], [9.0], [0.0])
    )
    recorder = ConditionsRecorder()
    tool = build_destination_conditions_tool(make_context(handler), recorder)

    output = await tool.ainvoke(
        {"destination": "Kyoto", "start_date": start, "end_date": start}
    )

    assert "Kyoto, Japan" in output
    assert recorder.location is not None
    assert recorder.outlook is not None
    assert recorder.outlook.source is WeatherSource.FORECAST


async def test_tool_degrades_to_prose_when_the_lookup_fails(make_context) -> None:
    tool = build_destination_conditions_tool(make_context(_router(geocoding={"results": []})))
    start = date.today() + timedelta(days=3)

    output = await tool.ainvoke(
        {"destination": "Atlantis", "start_date": start, "end_date": start}
    )

    assert "could not retrieve conditions" in output.lower()
