"""Shared fixtures.

Two rules shape everything here. Settings are always built with ``_env_file=None`` so
a developer's real ``.env`` cannot leak into a test run and change its outcome. And
every outbound call goes through an :class:`httpx.MockTransport`, so the suite never
opens a socket and never spends free-tier quota.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from itinera.api.app import create_app
from itinera.core.cache import TwoLayerCache
from itinera.core.config import LLMProviderName, Settings
from itinera.core.http import HttpClient
from itinera.core.resilience import CircuitBreakerRegistry
from itinera.domain.travel import TravelRequest
from itinera.llm.providers.fake import FakeChatModel
from itinera.services.runtime import Runtime
from itinera.tools.base import ToolContext

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return configuration isolated from the environment and the developer's disk.

    ``llm_provider_chain`` is pinned to the offline provider so no test can reach a
    real model, whatever keys happen to be present in the developer's environment.
    """
    return Settings(
        _env_file=None,
        cache_dir=tmp_path / "cache",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'itinera.db').as_posix()}",
        http_max_retries=0,
        llm_provider_chain="fake",
        serper_api_key="test-serper-key",
        exa_api_key="test-exa-key",
    )


@pytest.fixture
def start_date() -> date:
    """Return a trip start far enough out to stay valid whenever the suite runs."""
    return date.today() + timedelta(days=45)


@pytest.fixture
def travel_request(start_date: date) -> TravelRequest:
    """Return a representative four-day, two-traveller request."""
    return TravelRequest(
        destination="Kyoto, Japan",
        start_date=start_date,
        end_date=start_date + timedelta(days=3),
        budget_min=1500.0,
        budget_max=2500.0,
        currency="USD",
        interests=["food", "temples"],
        travelers=2,
    )


@pytest.fixture
def make_context(settings: Settings) -> Iterator[Callable[[Handler], ToolContext]]:
    """Return a factory building a tool context whose HTTP calls hit ``handler``.

    Contexts are closed at teardown so the disk cache releases its file handles, which
    Windows requires before the temporary directory can be removed.
    """
    contexts: list[ToolContext] = []

    def factory(handler: Handler) -> ToolContext:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        context = ToolContext(
            settings=settings,
            http=HttpClient(settings, client=client),
            cache=TwoLayerCache(settings),
            breakers=CircuitBreakerRegistry(
                failure_threshold=settings.circuit_breaker_failure_threshold,
                cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
            ),
        )
        contexts.append(context)
        return context

    yield factory

    for context in contexts:
        context.cache.close()


@pytest.fixture
def json_handler() -> Callable[..., Handler]:
    """Return a factory for handlers replying with a fixed JSON body.

    The returned handler records every request it served on its ``calls`` attribute,
    which is how tests assert that a second call was served from the cache.
    """

    def factory(payload: object, status_code: int = 200) -> Handler:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(status_code, json=payload)

        handler.calls = calls  # type: ignore[attr-defined]
        return handler

    return factory


@pytest.fixture
def runtime(settings: Settings) -> Runtime:
    """Return an unstarted runtime backed by temporary files and the offline provider.

    Unstarted because the two ways of driving the workflow start it differently: a
    graph-level test starts it directly, while an API test lets the application's own
    lifespan do it. Both share this one object, so a test can script the model through
    the same registry the workflow will use.
    """
    return Runtime(settings)


@pytest.fixture
def runtime_factory(settings: Settings) -> Callable[..., Runtime]:
    """Return a factory building an unstarted runtime with configuration overrides.

    For the cases where the behaviour under test *is* a configuration value, such as
    the revision ceiling.
    """

    def factory(**overrides: Any) -> Runtime:
        return Runtime(settings.model_copy(update=overrides))

    return factory


@pytest_asyncio.fixture
async def started_runtime(runtime: Runtime) -> AsyncIterator[Runtime]:
    """Return a runtime with its databases open and its graph compiled."""
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.stop()


@pytest.fixture
def fake_model(runtime: Runtime) -> FakeChatModel:
    """Return the offline chat model, so a test can script what it returns."""
    model = runtime.registry.chat_model(LLMProviderName.FAKE)
    assert isinstance(model, FakeChatModel)
    return model


@pytest_asyncio.fixture
async def client(settings: Settings, runtime: Runtime) -> AsyncIterator[AsyncClient]:
    """Return an HTTP client bound to the application.

    The application's real lifespan runs, so start-up and shutdown are exercised rather
    than bypassed, and the runtime it uses is the same one the test holds.
    """
    app = create_app(settings, runtime=runtime)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://itinera.test") as http:
            yield http


@pytest.fixture
def plan_payload(travel_request: TravelRequest) -> dict[str, Any]:
    """Return the JSON body for the sample travel request."""
    return {
        "destination": travel_request.destination,
        "start_date": travel_request.start_date.isoformat(),
        "end_date": travel_request.end_date.isoformat(),
        "budget_min": travel_request.budget_min,
        "budget_max": travel_request.budget_max,
        "currency": travel_request.currency,
        "interests": travel_request.interests,
        "travelers": travel_request.travelers,
    }
