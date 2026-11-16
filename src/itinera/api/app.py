"""The application factory.

A factory rather than a module-level app, because the runtime it depends on has to be
constructible from something other than the process environment - a test builds one
pointed at temporary files and the offline provider and mounts the same routes against
it.

The lifespan owns start-up and shutdown. Everything the process opens is opened there
and closed there, so a run that ends leaves no connection behind.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from itinera.api.errors import register_error_handlers
from itinera.api.routes import health, plans
from itinera.core.config import Settings, get_settings
from itinera.core.logging import configure_logging, get_logger
from itinera.core.trust import enable_system_trust_store
from itinera.services.planning import PlanningService
from itinera.services.runtime import Runtime

logger = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

_TITLE = "AI Trip Planner"
_VERSION = "0.1.0"
_DESCRIPTION = """
Multi-agent travel planning with human-in-the-loop approval.

Submit a travel request to `POST /plan`. A research agent gathers destination
intelligence from live sources, a planner agent drafts a day-by-day itinerary, and the
workflow then **pauses**: the draft is checkpointed and surfaced for review rather than
finalised. Poll `GET /plan/{id}` until the status reaches `awaiting_review`, then
approve, reject with feedback, or modify it through `POST /plan/{id}/review`. The
approved plan, with packing and preparation lists, is served from `GET /plan/{id}/final`.

Because the pause is checkpointed rather than held open, the review may arrive minutes
later, from a different client, or after a restart.
"""


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Configuration to use; defaults to the process environment.
        runtime: A pre-built runtime, injected by tests. When supplied, the lifespan
            still starts and stops it, so the test exercises the real start-up path.

    Returns:
        The configured application.
    """
    active_settings = settings or get_settings()
    configure_logging(active_settings)
    enable_system_trust_store(active_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start the runtime, serve, then release everything it opened."""
        active_runtime = runtime or Runtime(active_settings)
        await active_runtime.start()

        app.state.runtime = active_runtime
        app.state.planning_service = PlanningService(active_runtime)
        try:
            yield
        finally:
            await active_runtime.stop()

    app = FastAPI(
        title=_TITLE,
        version=_VERSION,
        description=_DESCRIPTION.strip(),
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(plans.router)
    _mount_console(app)

    logger.info("api.created", environment=active_settings.app_env.value)
    return app


def _mount_console(app: FastAPI) -> None:
    """Serve the browser console that drives the same four endpoints.

    The console is a client of the API, not a privileged path into it: it holds no
    state the server does not, and every action it takes is a request a reviewer could
    equally make from the OpenAPI page. Mounting it is therefore optional - if the
    directory is absent, the API still serves normally.
    """
    if not _STATIC_DIR.is_dir():  # pragma: no cover - only when packaged without assets
        return

    @app.get("/", include_in_schema=False)
    async def console() -> RedirectResponse:
        """Send the site root to the console."""
        return RedirectResponse(url="/ui/")

    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="console")
