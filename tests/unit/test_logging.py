"""Tests for log-stream setup.

The one thing worth pinning here is the encoding. A travel planner logs place names,
and most of the world's place names are not representable in a legacy Windows codepage.
"""

from __future__ import annotations

import io
import sys
from typing import Any

from itinera.core.config import Settings
from itinera.core.logging import _ensure_utf8_stdout, configure_logging, get_logger


class _Proxy:
    """Stands in for the colour-support wrapper, which forwards writes but not much else."""

    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped

    def write(self, text: str) -> int:
        return self.wrapped.write(text)


def _latin1_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")


def test_a_legacy_console_is_switched_to_utf8(monkeypatch) -> None:
    stream = _latin1_stream()
    monkeypatch.setattr(sys, "stdout", stream)

    _ensure_utf8_stdout()

    assert stream.encoding.lower().replace("-", "") == "utf8"


def test_the_stream_under_a_colour_wrapper_is_reached(monkeypatch) -> None:
    """The wrapper exposes no reconfigure of its own, so the real stream must be found."""
    stream = _latin1_stream()
    monkeypatch.setattr(sys, "stdout", _Proxy(stream))

    _ensure_utf8_stdout()

    assert stream.encoding.lower().replace("-", "") == "utf8"


def test_a_stream_that_cannot_be_reconfigured_is_left_alone(monkeypatch) -> None:
    """A redirected or detached stream must not turn logging setup into a crash."""
    monkeypatch.setattr(sys, "stdout", _Proxy(object()))

    _ensure_utf8_stdout()


def test_a_destination_that_needs_more_than_latin1_still_logs(monkeypatch, capsys) -> None:
    """The regression this exists for: 'Tōfuku-ji' used to take down the request."""
    stream = _latin1_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    configure_logging(Settings(_env_file=None))  # type: ignore[call-arg]

    get_logger(__name__).info("workflow.validated", destination="Tōfuku-ji, 京都")

    stream.flush()
    written = stream.buffer.getvalue().decode("utf-8")  # type: ignore[attr-defined]
    assert "Tōfuku-ji" in written
