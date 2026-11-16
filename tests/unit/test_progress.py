"""Tests for the in-flight progress tracker.

The tracker exists so a long phase is legible while it runs. Its contract is narrow but
strict: it must never raise into a run, and it must not grow without bound.
"""

from __future__ import annotations

from itinera.services.progress import ProgressTracker, note, running, tracker


def test_notes_reach_the_plan_being_run() -> None:
    with running("plan-a"):
        note("research", "web search: temples")
        note("research", "destination conditions: Kyoto")

    snapshot = tracker.snapshot("plan-a")

    assert snapshot is not None
    assert [n["message"] for n in snapshot["notes"]] == [
        "web search: temples",
        "destination conditions: Kyoto",
    ]
    assert all(n["phase"] == "research" for n in snapshot["notes"])


def test_a_note_outside_a_run_is_discarded_rather_than_raising() -> None:
    """Progress reporting must never be the reason a run fails."""
    note("research", "nobody is listening")


def test_an_unknown_plan_has_no_snapshot() -> None:
    assert ProgressTracker().snapshot("never-started") is None


def test_notes_are_bounded() -> None:
    local = ProgressTracker()
    local.begin("plan-b")

    for index in range(200):
        local.note("planner", f"note {index}", plan_id="plan-b")

    snapshot = local.snapshot("plan-b")
    assert snapshot is not None
    assert len(snapshot["notes"]) <= 40
    assert snapshot["notes"][-1]["message"] == "note 199", "the newest must survive"


def test_old_runs_are_evicted() -> None:
    local = ProgressTracker()

    for index in range(80):
        local.begin(f"plan-{index}")

    assert local.snapshot("plan-0") is None, "the oldest run should have been dropped"
    assert local.snapshot("plan-79") is not None, "the newest run must remain"


def test_beginning_a_run_clears_what_came_before() -> None:
    """A revision is a new run of the same plan, not a continuation of the old one."""
    local = ProgressTracker()
    local.begin("plan-c")
    local.note("research", "first run", plan_id="plan-c")

    local.begin("plan-c")

    snapshot = local.snapshot("plan-c")
    assert snapshot is not None
    assert snapshot["notes"] == []


def test_elapsed_is_measured_from_the_start_of_the_run() -> None:
    local = ProgressTracker()
    local.begin("plan-d")
    local.note("research", "something", plan_id="plan-d")

    snapshot = local.snapshot("plan-d")
    assert snapshot is not None
    assert snapshot["elapsed_seconds"] >= 0
    assert snapshot["notes"][0]["elapsed_seconds"] >= 0


def test_forgetting_a_plan_removes_it() -> None:
    local = ProgressTracker()
    local.begin("plan-e")

    local.forget("plan-e")

    assert local.snapshot("plan-e") is None
