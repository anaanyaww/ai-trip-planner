"""Live progress for a run that is still in flight.

A plan's audit trail is durable but coarse: entries reach it only when the workflow
checkpoints, and the research phase is a single checkpoint that can take minutes. To
someone watching, a correct system and a hung one look identical for that whole window.

This module carries the finer signal - which search is running, which day is being
checked, which candidate answered - from inside the agents to whoever is polling. It is
deliberately *not* persisted. Progress describes a run happening right now in this
process, and a run does not survive the process anyway (``BackgroundTasks`` ties it to
the worker that accepted it), so writing each tool call to the database would buy
durability the surrounding design cannot honour, at the cost of a write per call.

The tracker is therefore in-memory, bounded, and best-effort. If it is empty the client
falls back to the audit trail, which is the durable record.
"""

from __future__ import annotations

import time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

_MAX_NOTES = 40

_MAX_PLANS = 32

_current_plan: ContextVar[str | None] = ContextVar("current_plan", default=None)


@dataclass(frozen=True)
class Note:
    """One thing that happened, as it happened."""

    phase: str
    message: str
    at: float

    def as_dict(self, origin: float) -> dict[str, Any]:
        """Render for the API, with seconds elapsed rather than a wall-clock time."""
        return {
            "phase": self.phase,
            "message": self.message,
            "elapsed_seconds": round(max(self.at - origin, 0.0), 1),
        }


@dataclass
class _Run:
    """The notes for one plan, and when its clock started."""

    started_at: float = field(default_factory=time.monotonic)
    notes: deque[Note] = field(default_factory=lambda: deque(maxlen=_MAX_NOTES))


class ProgressTracker:
    """Collects in-flight notes per plan.

    Every method is safe to call for a plan that was never started and safe to call
    from any thread: progress reporting must never be the reason a run fails.
    """

    def __init__(self) -> None:
        """Start with no runs in flight."""
        self._runs: dict[str, _Run] = {}
        self._lock = Lock()

    def begin(self, plan_id: str) -> None:
        """Start (or restart) the clock for a plan and drop any earlier notes."""
        with self._lock:
            self._runs[plan_id] = _Run()
            self._evict()

    def note(self, phase: str, message: str, plan_id: str | None = None) -> None:
        """Record one note against the plan currently being run."""
        target = plan_id or _current_plan.get()
        if target is None:
            return
        with self._lock:
            run = self._runs.get(target)
            if run is None:
                run = self._runs.setdefault(target, _Run())
                self._evict()
            run.notes.append(Note(phase=phase, message=message, at=time.monotonic()))

    def snapshot(self, plan_id: str) -> dict[str, Any] | None:
        """Return what is known about a run, or ``None`` if nothing is."""
        with self._lock:
            run = self._runs.get(plan_id)
            if run is None:
                return None
            notes = list(run.notes)
            started = run.started_at

        return {
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "notes": [note.as_dict(started) for note in notes],
        }

    def forget(self, plan_id: str) -> None:
        """Drop a plan's notes once nobody will ask for them again."""
        with self._lock:
            self._runs.pop(plan_id, None)

    def _evict(self) -> None:
        """Drop the oldest runs once the map is full. Caller holds the lock."""
        while len(self._runs) > _MAX_PLANS:
            oldest = min(self._runs, key=lambda key: self._runs[key].started_at)
            del self._runs[oldest]


tracker = ProgressTracker()


def note(phase: str, message: str) -> None:
    """Record a note against the plan being run on this task.

    A module-level function rather than a passed-around dependency, because the callers
    are tools and agent loops whose signatures are shaped by what the *model* needs to
    see. Threading a reporter through them would put an observability concern into the
    contract every tool has to satisfy.
    """
    tracker.note(phase, message)


class running:  # noqa: N801 - used as a context manager, reads as a verb
    """Marks a block as belonging to one plan, so notes find their way home."""

    def __init__(self, plan_id: str) -> None:
        """Name the plan whose notes this block should collect."""
        self._plan_id = plan_id
        self._token: Any = None

    def __enter__(self) -> ProgressTracker:
        """Begin the run's clock and route notes to it."""
        tracker.begin(self._plan_id)
        self._token = _current_plan.set(self._plan_id)
        return tracker

    def __exit__(self, *_exc: object) -> None:
        """Stop routing notes here; the collected notes stay readable."""
        _current_plan.reset(self._token)
