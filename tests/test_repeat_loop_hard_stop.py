"""A model that re-issues the same rejected tool call must be STOPPED, not
nudged forever.

Observed 2026-09-08 (HarnessBench 087, 27B model): 26 consecutive identical
bash calls, each answered with "REJECTED: repeated identical tool pattern" plus
the strategy-change nudge, until the 20 minute task cap. Rejection + nudge is
feedback; after the model ignores several nudges the turn has to end with an
honest reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode.headless_json import _terminal_from_status
from tests.e2e.fake_runtime import build_test_app, tool_round
from tests.e2e.harness import EventRecorder, run_one_turn


def test_repeated_rejected_call_ends_the_turn(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "README.md").write_text("x\n")
    same = ("bash", {"command": "echo probing"})
    script = [tool_round(same) for _ in range(40)]  # would run 40 rounds if nothing stops it
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), "Fix the failing CLI parser test")

    assert trace.error is None
    assert getattr(app, "_last_turn_loop_exit_reason", "") == "repeat_loop_exhausted"
    # it stopped well before the script ran out
    assert len(app.engine.calls) < 25, len(app.engine.calls)
    status, code, _reason, text = _terminal_from_status(
        completion_status=str(getattr(app, "_last_turn_completion_status", "") or ""),
        blocked_reason="",
        loop_exit_reason=str(getattr(app, "_last_turn_loop_exit_reason", "") or ""),
        final_text=trace.final_response or "",
    )
    assert (status, code) == ("incomplete", 1)
    assert "kept re-issuing the same" in text
