"""Checkpoint serialization.

LangGraph's default serializer will deserialize any type it finds in a checkpoint,
warning as it goes. That is a real hazard rather than a nuisance: anything able to
write to the checkpoint database can name a type to reconstruct, so an allowlist is
the difference between reading data and executing whatever the row says.

The allowlist is derived from the domain package rather than typed out, so a model
added to the domain is checkpointable without anyone remembering to register it, and
nothing outside the domain is allowed by accident.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from inspect import getmembers, isclass
from types import ModuleType

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from itinera.core.logging import get_logger
from itinera.domain import enums, itinerary, research, review, travel

logger = get_logger(__name__)

_CHECKPOINTED_MODULES: tuple[ModuleType, ...] = (travel, research, itinerary, review, enums)


def _checkpointed_types() -> Iterator[type]:
    """Yield every domain model and enum that may appear in a checkpoint.

    Only classes actually defined in the domain modules are yielded; names those
    modules merely import would otherwise widen the allowlist to their dependencies.
    """
    for module in _CHECKPOINTED_MODULES:
        for _, member in getmembers(module, isclass):
            if member.__module__ != module.__name__:
                continue
            if issubclass(member, (BaseModel, Enum)):
                yield member


def build_serializer() -> JsonPlusSerializer:
    """Return a serializer that will only reconstruct this application's own types."""
    allowed = tuple(_checkpointed_types())
    logger.debug("checkpointer.allowlist_built", types=len(allowed))
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)
