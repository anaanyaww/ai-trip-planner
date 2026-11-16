"""Tests for the browser console mount.

The console is a client of the public API rather than a privileged path into it, so
what matters here is that mounting it changes nothing about the API itself.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from itinera.api.app import _STATIC_DIR


@pytest.mark.asyncio
async def test_the_site_root_leads_to_the_console(client: AsyncClient) -> None:
    response = await client.get("/", follow_redirects=False)

    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/ui/"


@pytest.mark.asyncio
async def test_the_console_is_served(client: AsyncClient) -> None:
    response = await client.get("/ui/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Trip Planner" in response.text


@pytest.mark.asyncio
async def test_the_console_assets_are_served(client: AsyncClient) -> None:
    for asset, kind in (("app.css", "css"), ("app.js", "javascript")):
        response = await client.get(f"/ui/{asset}")

        assert response.status_code == 200, asset
        assert kind in response.headers["content-type"], asset


@pytest.mark.asyncio
async def test_the_console_is_absent_from_the_api_schema(client: AsyncClient) -> None:
    """A reviewer reading the OpenAPI page should see endpoints, not page furniture."""
    paths = (await client.get("/openapi.json")).json()["paths"]

    assert "/" not in paths
    assert not any(path.startswith("/ui") for path in paths)


def test_every_asset_the_page_references_exists() -> None:
    """A missing stylesheet or script fails silently in a browser; catch it here."""
    markup = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    for asset in ("app.css", "app.js"):
        assert f"./{asset}" in markup, f"{asset} is not referenced"
        assert (_STATIC_DIR / asset).is_file(), f"{asset} is referenced but missing"


@pytest.mark.asyncio
async def test_a_running_plan_reports_live_progress(client: AsyncClient) -> None:
    """The status endpoint must carry the in-flight detail the console renders."""
    from itinera.services.progress import tracker

    created = (
        await client.post(
            "/plan",
            json={
                "destination": "Porto, Portugal",
                "start_date": "2099-05-01",
                "end_date": "2099-05-03",
                "budget_min": 700,
                "budget_max": 1100,
                "currency": "USD",
                "interests": ["riverside"],
                "travelers": 2,
            },
        )
    ).json()
    plan_id = created["plan_id"]
    tracker.begin(plan_id)
    tracker.note("research", "web search: Porto riverside", plan_id=plan_id)

    body = (await client.get(f"/plan/{plan_id}")).json()

    assert body["progress"]["notes"][-1]["message"] == "web search: Porto riverside"
    assert body["progress"]["elapsed_seconds"] >= 0
    tracker.forget(plan_id)


@pytest.mark.asyncio
async def test_a_plan_with_no_run_in_this_process_reports_no_progress(
    client: AsyncClient,
) -> None:
    """Progress is in-memory, so its absence must be ordinary rather than an error."""
    created = (
        await client.post(
            "/plan",
            json={
                "destination": "Porto, Portugal",
                "start_date": "2099-05-01",
                "end_date": "2099-05-03",
                "budget_min": 700,
                "budget_max": 1100,
                "currency": "USD",
                "interests": [],
                "travelers": 1,
            },
        )
    ).json()
    from itinera.services.progress import tracker

    tracker.forget(created["plan_id"])

    body = (await client.get(f"/plan/{created['plan_id']}")).json()

    assert body["progress"] is None
