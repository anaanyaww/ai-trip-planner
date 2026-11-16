"""Liveness and readiness endpoints.

Split deliberately. ``/health`` answers whether the process is up, which is what an
orchestrator restarts on. ``/ready`` answers whether it can actually serve a plan,
which depends on the database. Collapsing them would make a momentarily unreachable
database look like a crashed process.

Readiness does not require a model provider. The service degrades to the offline
provider rather than failing, so a missing key is reported as a warning in the body
rather than as an unready process.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from itinera.api.dependencies import RuntimeDep

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness")
async def health() -> dict[str, str]:
    """Return that the process is running."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness")
async def ready(runtime: RuntimeDep, response: Response) -> dict[str, Any]:
    """Report whether the service can serve a planning request."""
    database_ok = await runtime.database.healthcheck()
    if not database_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if database_ok else "degraded",
        "environment": runtime.settings.app_env.value,
        "database": "ok" if database_ok else "unavailable",
        "prompt_version": runtime.prompts.version,
        "model_providers": runtime.registry.describe(),
        "warnings": (
            ["No model provider is configured; plans will be placeholders."]
            if runtime.is_offline
            else []
        ),
    }
