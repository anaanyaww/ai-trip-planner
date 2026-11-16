"""Shared plumbing for agent tools.

Tools are built by factory functions that close over a :class:`ToolContext` rather
than importing a global client or settings object. That keeps each tool a pure
function of its inputs and its injected dependencies, which is what makes them
testable without patching module state.

:func:`guarded_call` applies the three protections every outbound call needs -
circuit breaker, cache and failure bookkeeping - in one place, so a new tool gets
them by using it rather than by remembering to reimplement them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from itinera.core.cache import TwoLayerCache, build_cache_key
from itinera.core.config import Settings
from itinera.core.exceptions import ExternalServiceError, ItineraError
from itinera.core.http import HttpClient
from itinera.core.logging import get_logger
from itinera.core.resilience import CircuitBreakerRegistry

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class ToolContext:
    """Dependencies shared by every tool."""

    settings: Settings
    http: HttpClient
    cache: TwoLayerCache
    breakers: CircuitBreakerRegistry

    @classmethod
    def create(cls, settings: Settings) -> ToolContext:
        """Build a context with freshly constructed collaborators."""
        return cls(
            settings=settings,
            http=HttpClient(settings),
            cache=TwoLayerCache(settings),
            breakers=CircuitBreakerRegistry(
                failure_threshold=settings.circuit_breaker_failure_threshold,
                cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
            ),
        )

    async def aclose(self) -> None:
        """Release the HTTP pool and the on-disk cache handles."""
        await self.http.aclose()
        self.cache.close()


async def guarded_call(
    context: ToolContext,
    *,
    service: str,
    cache_namespace: str,
    cache_params: dict[str, Any],
    ttl_seconds: int,
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Run ``operation`` behind the cache and the circuit breaker.

    The cache is consulted first, so a cached answer is served even while the breaker
    is open - a repeated question about a destination we already researched should not
    fail just because the provider is currently unhealthy.

    Args:
        context: Injected dependencies.
        service: Dependency name, used for the breaker, logs and errors.
        cache_namespace: Logical grouping for the cache key.
        cache_params: Parameters that fully determine the response.
        ttl_seconds: Lifetime of the cached entry.
        operation: The call to make on a cache miss.

    Returns:
        The cached or freshly fetched result.

    Raises:
        CircuitOpenError: If the dependency is in cooldown and nothing is cached.
        ExternalServiceError: If the call fails.
    """
    key = build_cache_key(cache_namespace, cache_params)

    cached = await context.cache.get(key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    breaker = context.breakers.get(service)
    breaker.ensure_available()

    try:
        result = await operation()
    except ItineraError:
        breaker.record_failure()
        raise
    except Exception as exc:  # noqa: BLE001 - normalised into the app's hierarchy below
        breaker.record_failure()
        logger.exception("tool.unexpected_error", service=service)
        raise ExternalServiceError(f"{service} call failed: {exc}", service=service) from exc

    breaker.record_success()
    await context.cache.set(key, result, ttl_seconds)
    return result
