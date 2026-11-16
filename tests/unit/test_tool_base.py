"""Tests for the shared tool plumbing.

``guarded_call`` is the single point through which every outbound tool call passes, so
its three behaviours - serve from cache, respect the breaker, normalise failures - are
worth pinning down here rather than re-testing through each tool.
"""

from __future__ import annotations

import httpx
import pytest

from itinera.core.exceptions import CircuitOpenError, ExternalServiceError, ToolExecutionError
from itinera.tools.base import ToolContext, guarded_call

CACHE_ARGS = {
    "service": "demo",
    "cache_namespace": "demo:op",
    "cache_params": {"q": "kyoto"},
    "ttl_seconds": 600,
}


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={})


async def test_second_call_is_served_from_cache(make_context) -> None:
    context: ToolContext = make_context(_ok)
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "fresh"

    first = await guarded_call(context, operation=operation, **CACHE_ARGS)
    second = await guarded_call(context, operation=operation, **CACHE_ARGS)

    assert (first, second) == ("fresh", "fresh")
    assert calls == 1


async def test_differing_parameters_do_not_share_an_entry(make_context) -> None:
    context: ToolContext = make_context(_ok)
    args = dict(CACHE_ARGS)
    args.pop("cache_params")

    async def operation_a() -> str:
        return "a"

    async def operation_b() -> str:
        return "b"

    first = await guarded_call(
        context, cache_params={"q": "kyoto"}, operation=operation_a, **args
    )
    second = await guarded_call(
        context, cache_params={"q": "osaka"}, operation=operation_b, **args
    )

    assert (first, second) == ("a", "b")


async def test_application_errors_pass_through_unchanged(make_context) -> None:
    context: ToolContext = make_context(_ok)

    async def operation() -> str:
        raise ToolExecutionError("upstream said no")

    with pytest.raises(ToolExecutionError, match="upstream said no"):
        await guarded_call(context, operation=operation, **CACHE_ARGS)


async def test_unexpected_errors_are_normalised(make_context) -> None:
    context: ToolContext = make_context(_ok)

    async def operation() -> str:
        raise RuntimeError("socket exploded")

    with pytest.raises(ExternalServiceError, match="socket exploded"):
        await guarded_call(context, operation=operation, **CACHE_ARGS)


async def test_breaker_opens_after_repeated_failures(make_context, settings) -> None:
    context: ToolContext = make_context(_ok)
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ToolExecutionError("still down")

    for index in range(settings.circuit_breaker_failure_threshold):
        with pytest.raises(ToolExecutionError):
            await guarded_call(
                context, cache_params={"q": index}, operation=operation, **_without_params()
            )

    with pytest.raises(CircuitOpenError):
        await guarded_call(
            context, cache_params={"q": "next"}, operation=operation, **_without_params()
        )

    assert attempts == settings.circuit_breaker_failure_threshold


async def test_cache_is_served_even_while_the_breaker_is_open(make_context, settings) -> None:
    """A question already answered should not fail because the provider is unhealthy."""
    context: ToolContext = make_context(_ok)

    async def succeed() -> str:
        return "cached answer"

    async def fail() -> str:
        raise ToolExecutionError("down")

    await guarded_call(context, operation=succeed, **CACHE_ARGS)

    for index in range(settings.circuit_breaker_failure_threshold):
        with pytest.raises(ToolExecutionError):
            await guarded_call(
                context, cache_params={"q": index}, operation=fail, **_without_params()
            )

    assert await guarded_call(context, operation=fail, **CACHE_ARGS) == "cached answer"


def _without_params() -> dict[str, object]:
    """Return the shared cache arguments minus the parameters each test varies."""
    args = dict(CACHE_ARGS)
    args.pop("cache_params")
    return args
