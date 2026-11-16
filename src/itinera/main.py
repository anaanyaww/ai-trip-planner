"""ASGI entry point.

Kept as a thin module so ``uvicorn itinera.main:app`` works while the application
itself stays constructible from a factory that tests can call with their own
configuration.
"""

from __future__ import annotations

from itinera.api.app import create_app

app = create_app()
