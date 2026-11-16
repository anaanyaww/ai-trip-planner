"""Translation of application errors into HTTP responses.

Every deliberate failure derives from :class:`ItineraError` and carries the status it
should produce, so one handler covers the whole hierarchy: a new failure mode is
mapped correctly without touching a route. Responses follow RFC 7807, so a client can
branch on ``type`` rather than pattern-matching prose.

The context attached to an exception stays in the logs. It exists to make an incident
diagnosable, and it may name internal services or configuration, so it is never
serialised into a response.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from itinera.core.exceptions import ItineraError, RateLimitedError
from itinera.core.logging import get_logger

logger = get_logger(__name__)

_SERVER_ERROR_THRESHOLD = 500
_GENERIC_DETAIL = "The request could not be completed. Please try again."
_UNPROCESSABLE = 422


def _title(code: str) -> str:
    """Render an error code as a human-readable title."""
    return code.replace("_", " ").title()


def _problem(status: int, code: str, detail: str) -> JSONResponse:
    """Build an RFC 7807 response body."""
    return JSONResponse(
        status_code=status,
        content={
            "type": code,
            "title": _title(code),
            "status": status,
            "detail": detail,
        },
        media_type="application/problem+json",
    )


async def handle_itinera_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate an application error into its declared status."""
    assert isinstance(exc, ItineraError)  # noqa: S101 - registered only for this type

    log = logger.error if exc.http_status >= _SERVER_ERROR_THRESHOLD else logger.info
    log(
        "api.error",
        code=exc.code,
        status=exc.http_status,
        path=request.url.path,
        detail=exc.detail,
        **exc.context,
    )

    response = _problem(exc.http_status, exc.code, exc.detail)
    if isinstance(exc, RateLimitedError) and exc.retry_after_seconds is not None:
        response.headers["Retry-After"] = str(int(exc.retry_after_seconds))
    return response


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Return 422 for a body that parses but does not satisfy the domain's rules.

    Cross-field rules - a trip ending before it starts, a budget ceiling below its
    floor - live on the domain models, so they are raised when the body is converted
    rather than when it is parsed. Without this the client would see a 500 for what is
    plainly a bad request.
    """
    assert isinstance(exc, ValidationError)  # noqa: S101 - registered only for this type

    reasons = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
        for error in exc.errors()
    )
    logger.info("api.validation_failed", path=request.url.path, reasons=reasons)
    return _problem(_UNPROCESSABLE, "validation_error", reasons)


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 500 for anything not deliberately raised.

    The detail is deliberately uninformative: an unexpected exception's message can
    carry a file path, a query or a key, none of which belong in a response body.
    """
    logger.exception("api.unhandled_error", path=request.url.path)
    return _problem(500, "internal_error", _GENERIC_DETAIL)


def register_error_handlers(app: FastAPI) -> None:
    """Attach the error handlers to ``app``."""
    app.add_exception_handler(ItineraError, handle_itinera_error)
    app.add_exception_handler(ValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
