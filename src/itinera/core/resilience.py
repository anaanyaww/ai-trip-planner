"""Resilience primitives for calls to rate-limited, best-effort free-tier services.

Three concerns are handled separately so they can be composed per dependency:

* :class:`AsyncRateLimiter` paces outbound calls so we stay under a published quota
  instead of discovering it through 429 responses.
* :class:`CircuitBreaker` stops hammering a dependency that is already failing, which
  is what lets the LLM chain skip a dead provider immediately rather than paying its
  full timeout on every request.
* :func:`retry_delays` yields exponential backoff with jitter, honouring an upstream
  ``Retry-After`` hint when one is supplied.
"""

from __future__ import annotations

import asyncio
import random  # noqa: S311 - jitter only, not used for anything security-sensitive
import time
from collections.abc import Iterator
from enum import Enum

from itinera.core.exceptions import CircuitOpenError
from itinera.core.logging import get_logger

logger = get_logger(__name__)

_SECONDS_PER_MINUTE = 60.0
_JITTER_RATIO = 0.25

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable_status(status_code: int) -> bool:
    """Whether an HTTP status warrants a retry rather than an immediate failure."""
    return status_code in RETRYABLE_STATUS_CODES


def retry_delays(
    attempts: int,
    base_seconds: float,
    max_seconds: float,
    retry_after_seconds: float | None = None,
) -> Iterator[float]:
    """Yield backoff delays for successive retries.

    Delays grow exponentially and carry positive-only jitter, which spreads retries
    from concurrent callers instead of synchronising them into a second burst.

    Args:
        attempts: Number of retries to schedule; a value of zero yields nothing.
        base_seconds: Delay before the first retry.
        max_seconds: Ceiling applied to every delay.
        retry_after_seconds: Upstream-advertised cooldown. When present it is used for
            the first delay, since the server knows its own reset window better than
            our backoff curve does.

    Yields:
        Delay in seconds before each retry attempt.
    """
    for attempt in range(attempts):
        if attempt == 0 and retry_after_seconds is not None:
            yield min(retry_after_seconds, max_seconds)
            continue
        delay = min(base_seconds * (2**attempt), max_seconds)
        yield delay + random.uniform(0.0, delay * _JITTER_RATIO)  # noqa: S311


class AsyncRateLimiter:
    """Token-bucket limiter pacing calls to a fixed requests-per-minute budget.

    Refills continuously rather than in fixed windows, so a burst is smoothed instead
    of being allowed at a window boundary and then stalling for the remainder.
    """

    def __init__(self, requests_per_minute: int, name: str = "default") -> None:
        """Initialise the bucket at full capacity.

        Args:
            requests_per_minute: Sustained call budget.
            name: Identifier used in log records.

        Raises:
            ValueError: If ``requests_per_minute`` is not positive.
        """
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")

        self._name = name
        self._capacity = float(requests_per_minute)
        self._refill_per_second = requests_per_minute / _SECONDS_PER_MINUTE
        self._tokens = float(requests_per_minute)
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._tokens = min(
                    self._capacity, self._tokens + elapsed * self._refill_per_second
                )
                self._updated_at = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                wait_for = (1.0 - self._tokens) / self._refill_per_second

            logger.debug("rate_limit.wait", limiter=self._name, seconds=round(wait_for, 3))
            await asyncio.sleep(wait_for)


class CircuitState(str, Enum):
    """Lifecycle state of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Trips after consecutive failures and recovers through a single trial call.

    While open, calls are rejected immediately. Once the cooldown elapses the breaker
    admits one probe: success closes it, failure re-opens it for another cooldown.
    """

    def __init__(self, name: str, failure_threshold: int, cooldown_seconds: float) -> None:
        """Initialise a closed breaker.

        Args:
            name: Identifier of the guarded dependency, used in errors and logs.
            failure_threshold: Consecutive failures required to trip the breaker.
            cooldown_seconds: Time the breaker stays open before admitting a probe.

        Raises:
            ValueError: If the threshold is not positive.
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")

        self.name = name
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Current state, accounting for an elapsed cooldown."""
        if self._opened_at is None:
            return CircuitState.CLOSED
        if time.monotonic() - self._opened_at >= self._cooldown:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def is_available(self) -> bool:
        """Whether a call may proceed right now."""
        return self.state is not CircuitState.OPEN

    def ensure_available(self) -> None:
        """Raise if the breaker is open.

        Raises:
            CircuitOpenError: When the dependency is in cooldown.
        """
        if self.state is CircuitState.OPEN:
            assert self._opened_at is not None  # noqa: S101 - guarded by state check
            remaining = self._cooldown - (time.monotonic() - self._opened_at)
            raise CircuitOpenError(
                f"{self.name} is in cooldown after {self._failures} consecutive failures; "
                f"retrying in {remaining:.1f}s",
                provider=self.name,
                retry_in_seconds=round(remaining, 1),
            )

    def record_success(self) -> None:
        """Reset the breaker after a successful call."""
        if self._failures or self._opened_at is not None:
            logger.info("circuit.closed", provider=self.name)
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Register a failure, tripping the breaker once the threshold is reached."""
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()
            logger.warning(
                "circuit.opened",
                provider=self.name,
                failures=self._failures,
                cooldown_seconds=self._cooldown,
            )


class CircuitBreakerRegistry:
    """Lazily creates and holds one breaker per named dependency."""

    def __init__(self, failure_threshold: int, cooldown_seconds: float) -> None:
        """Store the defaults applied to every breaker this registry creates."""
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        """Return the breaker for ``name``, creating it on first use."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, self._threshold, self._cooldown)
        return self._breakers[name]

    def states(self) -> dict[str, str]:
        """Return a snapshot of every breaker's state, for health reporting."""
        return {name: breaker.state.value for name, breaker in self._breakers.items()}
