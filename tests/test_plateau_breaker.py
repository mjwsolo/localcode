"""A model that stops making progress must be told to wrap up, then stopped.

Observed 2026-09-08 (HarnessBench 043/086/087, 27B model): the oracle score at
the 20 minute cap equalled what other frontends got by STOPPING early — the
model had reached its ceiling and kept re-editing the same file / re-running
the same experiments instead of writing the remaining deliverables. Each edit
had different content and each command different output, so none of the
byte-identical breakers fired.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode.agent.plateau import PLATEAU_GRACE_ROUNDS, PLATEAU_NUDGE_AFTER, PlateauTracker
from localcode.agent.turn_finalization import status_for_exit
from localcode.headless_json import _terminal_from_status
from tests.e2e.fake_runtime import build_test_app, say, tool_round
from tests.e2e.harness import EventRecorder, run_one_turn

PROMPT = "Build a small CLI that parses CSV files and add a README"
NUDGE_MARK = "No new progress for"


def _nudge_reached_model(app) -> bool:
    return any(
        isinstance(m.get("content"), str) and NUDGE_MARK in m["content"]
        for call in app.engine.calls
        for m in call
    )


_N = [0]


def _round(*calls, text: str = "") -> list[dict]:
    """tool_round + the `stream_done` the real gateway emits. Real models narrate
    a different sentence each round; varying the prose length keeps the unrelated
    same-(content_chars, tool) signature breaker out of the way so these tests
    exercise the plateau detector alone."""
    _N[0] += 1
    text = text or ("Refining " + "." * (_N[0] % 7 + 1))
    events = tool_round(*calls, text=text)
    events.append({"type": "stream_done", "finish_reason": "tool_calls",
                   "content_chars": len(text)})
    return events


def _write(path: str, body: str):
    return _round(("write_file", {"path": path, "content": body}))


def test_same_file_churn_gets_wrap_up_nudge_then_stops(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    emitted: list[tuple[str, dict]] = []
    import localcode.events as _events
    monkeypatch.setattr(_events, "emit", lambda et, **f: emitted.append((et, f)))
    # Round 0 writes a real file (progress); rounds 1..20 rewrite the SAME file
    # with DIFFERENT content each time — the shape no other breaker catches.
    script = [_write("app.py", "print('v0')\n")]
    script += [_write("app.py", f"print('v{i}')\n") for i in range(1, 21)]
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), PROMPT)

    assert trace.error is None
    assert getattr(app, "_last_turn_loop_exit_reason", "") == "plateau_exhausted"
    assert _nudge_reached_model(app), "wrap-up nudge never reached the model"
    # Nudge fires after 6 no-progress rounds (never inside the grace window);
    # the stop fires 6 rounds later — well before the 21-round script runs out.
    nudge_round = next(
        i for i, call in enumerate(app.engine.calls)
        if any(isinstance(m.get("content"), str) and NUDGE_MARK in m["content"] for m in call)
    )
    assert nudge_round >= PLATEAU_GRACE_ROUNDS
    assert nudge_round <= PLATEAU_NUDGE_AFTER + PLATEAU_GRACE_ROUNDS + 1
    assert len(app.engine.calls) < 20, len(app.engine.calls)
    # The nudge was ephemeral — it must not survive into persisted history.
    assert not any(NUDGE_MARK in str(m.get("content", "")) for m in app.session.messages)

    # Honest final text: what was delivered, and that progress stopped.
    text = trace.final_response or ""
    assert "app.py" in text
    assert "no new progress" in text.lower()
    assert "incomplete" in text.lower()

    # Maps to status incomplete, exactly like stall_exhausted.
    assert status_for_exit("plateau_exhausted") == status_for_exit("stall_exhausted") == ("incomplete", "failed")
    status, code, _reason, _text = _terminal_from_status(
        completion_status=str(getattr(app, "_last_turn_completion_status", "") or ""),
        blocked_reason="",
        loop_exit_reason=str(getattr(app, "_last_turn_loop_exit_reason", "") or ""),
        final_text=text,
    )
    assert (status, code) == ("incomplete", 1)

    # Telemetry: one auto_nudge for the wrap-up, one loop_break for the stop.
    signals = [(et, f.get("signal")) for et, f in emitted if et in {"auto_nudge", "loop_break"}]
    assert ("auto_nudge", "plateau_wrap_up") in signals
    assert ("loop_break", "plateau_exhausted") in signals
    assert signals.index(("auto_nudge", "plateau_wrap_up")) < signals.index(("loop_break", "plateau_exhausted"))


def test_new_file_every_round_never_plateaus(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    script = [_write(f"src/mod_{i}.py", f"X = {i}\n") for i in range(15)]
    script.append(say("Done: wrote 15 modules."))
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), PROMPT)

    assert trace.error is None
    assert not _nudge_reached_model(app)
    assert getattr(app, "_last_turn_loop_exit_reason", "") != "plateau_exhausted"
    assert len(app.engine.calls) >= 16


def test_failing_then_passing_test_run_counts_as_progress(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    failing = _round(("bash", {"command": "echo 'FAILED tests/test_x.py' && false  # pytest"}))
    passing = _round(("bash", {"command": "echo '1 passed'  # pytest"}))
    script = [_write("app.py", "print('v0')\n")]              # round 0: progress
    script.append(failing)                                    # round 1: no progress (fails)
    script += [_write("app.py", f"print('v{i}')\n") for i in range(1, 4)]  # rounds 2..4
    script.append(passing)                                    # round 5: fail → pass = progress, resets
    script += [_write("app.py", f"print('w{i}')\n") for i in range(5)]     # rounds 6..10: 5 no-progress
    script.append(say("Done."))                               # round 11
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), PROMPT)

    assert trace.error is None
    # Without the reset the counter would have hit 6 at round 10 and nudged.
    assert not _nudge_reached_model(app)
    assert getattr(app, "_last_turn_loop_exit_reason", "") != "plateau_exhausted"
    assert len(app.engine.calls) >= 12

    # Control: the same script with a second FAILING run in place of the pass
    # is not progress, so the streak reaches 6 and the wrap-up nudge fires.
    repo2 = tmp_path / "project2"
    repo2.mkdir()
    control = list(script)
    control[5] = _round(("bash", {"command": "echo 'FAILED again' && false  # pytest"}))
    app2 = build_test_app(tmp_path / "h2", script=control, cwd=repo2)
    trace2 = run_one_turn(app2, EventRecorder(), PROMPT)
    assert trace2.error is None
    assert _nudge_reached_model(app2)


def test_tracker_unit_semantics():
    t = PlateauTracker(nudge_after=2, grace_rounds=0)
    kw = dict(build_outcomes=[], todos_before=(0, 0), todos_after=(0, 0))
    assert t.observe_round(round_idx=0, changed_new_file=True, **kw) == ""
    assert t.observe_round(round_idx=1, changed_new_file=False, **kw) == ""
    assert t.observe_round(round_idx=2, changed_new_file=False, **kw) == "wrap_up"
    assert t.no_progress_rounds == 0  # reset after the nudge
    assert t.observe_round(round_idx=3, changed_new_file=False, **kw) == ""
    assert t.observe_round(round_idx=4, changed_new_file=False, **kw) == "exhausted"

    # build outcomes: first pass counts, pass-after-pass does not, fail→pass does
    t = PlateauTracker()
    assert t.round_made_progress(changed_new_file=False, build_outcomes=[True], todos_before=(0, 0), todos_after=(0, 0))
    assert not t.round_made_progress(changed_new_file=False, build_outcomes=[True], todos_before=(0, 0), todos_after=(0, 0))
    assert not t.round_made_progress(changed_new_file=False, build_outcomes=[False], todos_before=(0, 0), todos_after=(0, 0))
    assert t.round_made_progress(changed_new_file=False, build_outcomes=[True], todos_before=(0, 0), todos_after=(0, 0))
    # todo completion counts; a list that closes out (cleared to []) counts too
    assert t.round_made_progress(changed_new_file=False, build_outcomes=[], todos_before=(3, 0), todos_after=(2, 1))
    assert t.round_made_progress(changed_new_file=False, build_outcomes=[], todos_before=(1, 2), todos_after=(0, 0))
    assert not t.round_made_progress(changed_new_file=False, build_outcomes=[], todos_before=(2, 1), todos_after=(2, 1))

    # grace window: no firing in the first rounds even with a long streak
    t = PlateauTracker(nudge_after=2, grace_rounds=4)
    for i in range(4):
        assert t.observe_round(round_idx=i, changed_new_file=False, **kw) == ""
    assert t.observe_round(round_idx=4, changed_new_file=False, **kw) == "wrap_up"
