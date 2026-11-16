"""Request-scoped dependencies.

The runtime is created by the lifespan handler and stored on the application, not in a
module-level global. Routes reach it through these providers, which is what lets a test
mount the same routes against a runtime pointed at temporary files and the offline
provider.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from itinera.services.planning import PlanningService
from itinera.services.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    """Return the started runtime.

    Raises:
        RuntimeError: If the application was built without a lifespan.
    """
    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None:  # pragma: no cover - the lifespan always sets it
        raise RuntimeError("The application runtime is not available")
    return runtime


def get_planning_service(request: Request) -> PlanningService:
    """Return the process-wide planning service.

    Shared rather than per-request because it owns the per-plan locks that serialise
    concurrent work; a fresh instance per request would hand out fresh locks and
    serialise nothing.
    """
    service: PlanningService | None = getattr(request.app.state, "planning_service", None)
    if service is None:  # pragma: no cover - the lifespan always sets it
        raise RuntimeError("The planning service is not available")
    return service


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
PlanningServiceDep = Annotated[PlanningService, Depends(get_planning_service)]
