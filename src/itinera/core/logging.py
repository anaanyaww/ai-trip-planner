"""Structured logging setup.

Console rendering is used locally for readability; JSON is emitted elsewhere so logs
stay machine-parsable in aggregation. A plan identifier bound as a context variable
propagates automatically into every record emitted while a workflow runs, which is
what makes a single plan's journey through the graph traceable across background
tasks and HTTP requests.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from structlog.types import Processor

from itinera.core.config import LogFormat, Settings

_SENSITIVE_KEYS = frozenset({"api_key", "authorization", "x-api-key", "token", "secret"})
_REDACTED = "***redacted***"


def _ensure_utf8_stdout() -> None:
    """Make the log stream able to carry the place names the logs actually contain.

    A Windows console defaults to a legacy codepage - cp1252 in Western locales - which
    cannot encode most of the world's destinations. Logging a plan for Tofuku-ji spelled
    correctly, or for anywhere in Asia, then raises ``UnicodeEncodeError`` from inside
    the logging call and takes down the request that emitted it. A travel planner whose
    logs crash on non-English destinations is not usable, so the stream is reconfigured
    to UTF-8, falling back to replacement characters if even that is refused.
    """
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        target = stream
        for _ in range(4):
            if target is None or hasattr(target, "reconfigure"):
                break
            target = getattr(target, "wrapped", None)
        if target is None or not hasattr(target, "reconfigure"):
            continue
        with contextlib.suppress(ValueError, OSError):
            target.reconfigure(encoding="utf-8", errors="replace")


def _redact_secrets(
    _logger: Any, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Replace credential-bearing values so keys never reach the log sink.

    Secrets are already wrapped in ``SecretStr`` at the configuration boundary; this is
    the second line of defence for values assembled ad hoc, such as request headers.
    """
    for key in list(event_dict):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Configure structlog and the standard library root logger.

    Args:
        settings: Active configuration supplying the level and rendering format.
    """
    level = getattr(logging, settings.log_level)

    _ensure_utf8_stdout()
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_secrets,
    ]

    renderer: Processor = (
        structlog.dev.ConsoleRenderer(colors=settings.log_format is LogFormat.CONSOLE)
        if settings.log_format is LogFormat.CONSOLE
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``."""
    return structlog.stdlib.get_logger(name)


def bind_plan_context(plan_id: str, **extra: Any) -> None:
    """Bind a plan identifier to the logging context for the current task."""
    bind_contextvars(plan_id=plan_id, **extra)


def clear_plan_context() -> None:
    """Clear all context variables bound to the current task."""
    clear_contextvars()
