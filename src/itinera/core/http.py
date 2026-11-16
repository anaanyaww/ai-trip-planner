"""Shared async HTTP client for outbound tool calls.

Every tool talks to a third-party HTTP API, and each of them needs the same
behaviour: a bounded timeout, retries on transient failures, respect for an upstream
``Retry-After`` header, and translation of transport errors into the application's
exception hierarchy. Centralising that here keeps the tool modules focused on their
own request and response shapes.
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any

import httpx

from itinera.core.config import Settings
from itinera.core.exceptions import ExternalServiceError, RateLimitedError
from itinera.core.logging import get_logger
from itinera.core.resilience import is_retryable_status, retry_delays

logger = get_logger(__name__)

_MAX_ERROR_BODY_CHARS = 300


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Return the ``Retry-After`` delay in seconds, when present and numeric."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class HttpClient:
    """Thin retrying wrapper over :class:`httpx.AsyncClient`.

    One instance is shared for the process lifetime so connections are pooled rather
    than renegotiated per tool call.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        """Create the client.

        Args:
            settings: Supplies the timeout and retry policy.
            client: Pre-built transport, injected by tests to avoid real sockets.
        """
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.http_timeout_seconds),
            follow_redirects=True,
            headers={"User-Agent": "ai-trip-planner/0.1"},
        )

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        service: str = "upstream",
    ) -> Any:
        """Perform a request and return the decoded JSON body.

        Retries transient failures - connection errors, timeouts and the retryable
        status codes - up to the configured attempt limit. Non-retryable error
        statuses fail immediately, since repeating a malformed request wastes quota.

        Args:
            method: HTTP verb.
            url: Absolute request URL.
            params: Query-string parameters.
            json_body: Request body serialised as JSON.
            headers: Additional request headers.
            service: Dependency name used in logs and error messages.

        Returns:
            The decoded JSON payload.

        Raises:
            RateLimitedError: If the final attempt was rejected with HTTP 429.
            ExternalServiceError: If the service errored, timed out, or returned a
                body that is not valid JSON.
        """
        attempts = self._settings.http_max_retries
        last_error: Exception | None = None
        retry_after: float | None = None

        for attempt in range(attempts + 1):
            if attempt:
                delay = next(
                    retry_delays(
                        attempts=1,
                        base_seconds=self._settings.http_backoff_base_seconds
                        * (2 ** (attempt - 1)),
                        max_seconds=self._settings.http_backoff_max_seconds,
                        retry_after_seconds=retry_after,
                    )
                )
                logger.info(
                    "http.retry",
                    service=service,
                    attempt=attempt,
                    delay_seconds=round(delay, 2),
                )
                await self._sleep(delay)

            try:
                response = await self._client.request(
                    method, url, params=params, json=json_body, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                retry_after = None
                logger.warning("http.transport_error", service=service, error=str(exc))
                continue

            if response.is_success:
                return self._decode(response, service)

            retry_after = _parse_retry_after(response)
            if not is_retryable_status(response.status_code):
                raise ExternalServiceError(
                    f"{service} returned HTTP {response.status_code}: "
                    f"{response.text[:_MAX_ERROR_BODY_CHARS]}",
                    service=service,
                    status_code=response.status_code,
                )

            last_error = ExternalServiceError(
                f"{service} returned HTTP {response.status_code}",
                service=service,
                status_code=response.status_code,
            )
            if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                last_error = RateLimitedError(
                    f"{service} rate limit reached",
                    retry_after_seconds=retry_after,
                    service=service,
                )

        assert last_error is not None  # noqa: S101 - the loop always sets it before exiting
        if isinstance(last_error, RateLimitedError):
            raise last_error
        raise ExternalServiceError(
            f"{service} did not respond successfully after {attempts + 1} attempts: "
            f"{last_error}",
            service=service,
        ) from last_error

    @staticmethod
    def _decode(response: httpx.Response, service: str) -> Any:
        """Decode a successful response body as JSON."""
        try:
            return response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                f"{service} returned a body that is not valid JSON", service=service
            ) from exc

    @staticmethod
    async def _sleep(seconds: float) -> None:
        """Sleep between retry attempts; overridden in tests to keep them fast."""
        await asyncio.sleep(seconds)

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> HttpClient:
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the connection pool on exit."""
        await self.aclose()
