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

from localcode.agent.plateau import (
    PLATEAU_GRACE_ROUNDS,
    PLATEAU_NUDGE_AFTER,
    PLATEAU_STOP_AFTER,
    PlateauTracker,
    failure_signature,
)
from localcode.agent.turn_finalization import status_for_exit
from localcode.headless_json import _terminal_from_status
from tests.e2e.fake_runtime import build_test_app, say, tool_round
from tests.e2e.harness import EventRecorder, run_one_turn

PROMPT = "Build a small CLI that parses CSV files and add a README"
NUDGE_MARK = "No new progress for"
FINISH_MARK = "Your check passed"


def _nudge_reached_model(app, mark: str = NUDGE_MARK) -> bool:
    return any(
        isinstance(m.get("content"), str) and mark in m["content"]
        for call in app.engine.calls
        for m in call
    )


def _nudge_texts(app, mark: str = NUDGE_MARK) -> list[str]:
    seen: list[str] = []
    for call in app.engine.calls:
        for m in call:
            c = m.get("content")
            if isinstance(c, str) and mark in c and c not in seen:
                seen.append(c)
    return seen


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


def _pytest_run(*failures: str):
    """A scripted `pytest` whose output lists the given FAILED lines (exit 1),
    or `N passed` (exit 0) when there are none. The trailing `# pytest` comment
    is what `ran_build_or_test` keys on."""
    if not failures:
        return _round(("bash", {"command": "echo '3 passed in 0.12s'  # pytest"}))
    body = "\n".join(f"FAILED tests/test_app.py::{name} - AssertionError: {name} broke" for name in failures)
    return _round(("bash", {"command": f"printf '{body}\\n{len(failures)} failed in 0.31s\\n' && false  # pytest"}))


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
    # the stop fires 8 rounds later — well before the 21-round script runs out.
    nudge_round = next(
        i for i, call in enumerate(app.engine.calls)
        if any(isinstance(m.get("content"), str) and NUDGE_MARK in m["content"] for m in call)
    )
    assert nudge_round >= PLATEAU_GRACE_ROUNDS
    assert nudge_round <= PLATEAU_NUDGE_AFTER + PLATEAU_GRACE_ROUNDS + 1
    assert len(app.engine.calls) <= nudge_round + PLATEAU_STOP_AFTER + 1, len(app.engine.calls)
    assert not _nudge_reached_model(app, FINISH_MARK)  # no check ever passed
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

    # Control: the same script with the SAME failing run in place of the pass
    # is not progress, so the streak reaches 6 and the wrap-up nudge fires.
    # (A run failing DIFFERENTLY would be progress — see the debugging-loop
    # test below — so the control must repeat the identical failure.)
    repo2 = tmp_path / "project2"
    repo2.mkdir()
    control = list(script)
    control[5] = _round(("bash", {"command": "echo 'FAILED tests/test_x.py' && false  # pytest"}))
    app2 = build_test_app(tmp_path / "h2", script=control, cwd=repo2)
    trace2 = run_one_turn(app2, EventRecorder(), PROMPT)
    assert trace2.error is None
    assert _nudge_reached_model(app2)


def test_tracker_unit_semantics():
    t = PlateauTracker(nudge_after=2, stop_after=3, grace_rounds=0)
    kw = dict(build_outcomes=[], todos_before=(0, 0), todos_after=(0, 0))
    assert t.observe_round(round_idx=0, changed_new_file=True, **kw) == ""
    assert t.observe_round(round_idx=1, changed_new_file=False, **kw) == ""
    assert t.observe_round(round_idx=2, changed_new_file=False, **kw) == "wrap_up"
    assert t.no_progress_rounds == 0  # reset after the nudge
    assert t.observe_round(round_idx=3, changed_new_file=False, **kw) == ""
    assert t.observe_round(round_idx=4, changed_new_file=False, **kw) == ""  # stop_after > nudge_after
    assert t.observe_round(round_idx=5, changed_new_file=False, **kw) == "exhausted"
    assert PLATEAU_STOP_AFTER > PLATEAU_NUDGE_AFTER

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

    # failure-set semantics: the FIRST failure is a baseline, a shrinking set is
    # progress, a new signature is progress ONCE, a signature seen before is not.
    t = PlateauTracker()
    ab = "FAILED t.py::a - AssertionError\nFAILED t.py::b - AssertionError\n2 failed in 0.3s"
    a = "FAILED t.py::a - AssertionError\n1 failed in 0.2s"
    c = "FAILED t.py::c - AssertionError\n1 failed in 0.2s"
    run = lambda out: t.round_made_progress(changed_new_file=False, build_outcomes=[False], build_outputs=[out], todos_before=(0, 0), todos_after=(0, 0))
    assert not run(ab)          # baseline
    assert not run(ab)          # identical
    assert run(a)               # 2 → 1 failure: shrank
    assert run(c)               # same count, unseen signature
    assert not run(a)           # seen before this turn: the true loop
    assert not run(c)
    assert "failures_shrank" in t.progress_log and "new_failure_set" in t.progress_log

    # normalisation: paths' directories, numbers and ANSI do not change identity
    assert failure_signature("FAILED /home/u/proj/tests/t.py::a - took 0.31s") == failure_signature(
        "\x1b[31mFAILED\x1b[0m tests/t.py::a - took 12.9s"
    )
    assert failure_signature("src/a.ts(12,3): error TS2322: x\nFound 1 error.") == frozenset({"a.ts(N,N): error TSN: x"})

    # recent pass at the stop threshold → ONE finish_now nudge, then more rounds
    t = PlateauTracker(nudge_after=1, stop_after=2, grace_rounds=0)
    assert t.observe_round(round_idx=0, changed_new_file=False, **kw) == "wrap_up"
    assert t.observe_round(round_idx=1, changed_new_file=False, build_outcomes=[True], todos_before=(0, 0), todos_after=(0, 0)) == ""  # first pass = progress
    assert t.observe_round(round_idx=2, changed_new_file=False, build_outcomes=[True], todos_before=(0, 0), todos_after=(0, 0)) == ""  # pass-after-pass: no progress
    assert t.observe_round(round_idx=3, changed_new_file=False, **kw) == "finish_now"  # pass 1 round ago
    assert t.finish_nudged
    for i in range(4, 4 + t.finish_rounds - 1):
        assert t.observe_round(round_idx=i, changed_new_file=False, **kw) == ""
    assert t.observe_round(round_idx=4 + t.finish_rounds - 1, changed_new_file=False, **kw) == "exhausted"

    # nudge text names the open plan items; generic wording without them
    assert "Open plan items:" not in PlateauTracker().wrap_up_nudge([])
    txt = PlateauTracker().wrap_up_nudge(["Create rollback.sql", "Write README"])
    assert "Open plan items:\n- Create rollback.sql\n- Write README" in txt
    assert "finish now" in PlateauTracker().finish_now_nudge()
    summ = PlateauTracker().exhausted_summary(changed_files=["a.py"], open_todos=["Create rollback.sql"])
    assert "Open plan items:\n- Create rollback.sql" in summ


# ── refinements (HarnessBench 2026-09-10: false positives to avoid) ──────────

def _plan_round(*items: tuple[str, str]):
    return _round(("todo_write", {"todos": [{"content": c, "status": st} for c, st in items]}))


def test_debugging_loop_with_changing_failures_is_never_nudged(tmp_path):
    """Re-editing ONE file while each pytest run fails DIFFERENTLY (3 → 2 → 1
    failures) is a genuine debugging loop, not a plateau."""
    repo = tmp_path / "project"
    repo.mkdir()
    script = [_write("app.py", "v0\n")]                       # round 0: new file
    script.append(_pytest_run("test_a", "test_b", "test_c"))  # round 1: baseline (3 failures)
    script.append(_write("app.py", "v1\n"))                   # round 2: re-edit
    script.append(_pytest_run("test_a", "test_b"))            # round 3: 2 failures → progress
    script.append(_write("app.py", "v2\n"))                   # round 4
    script.append(_pytest_run("test_a"))                      # round 5: 1 failure → progress
    script.append(_write("app.py", "v3\n"))                   # round 6
    script.append(_pytest_run("test_d"))                      # round 7: same count, NEW failure → progress
    script += [_write("app.py", f"v{i}\n") for i in range(4, 8)]  # rounds 8..11: 4 no-progress
    script.append(_pytest_run())                              # round 12: fail → pass = progress
    script.append(say("Done: tests pass."))
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), PROMPT)

    assert trace.error is None
    assert not _nudge_reached_model(app), _nudge_texts(app)
    assert getattr(app, "_last_turn_loop_exit_reason", "") != "plateau_exhausted"
    assert len(app.engine.calls) >= len(script)


def test_identical_failing_output_every_round_still_gets_nudge(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    script = [_write("app.py", "v0\n")]
    for i in range(1, 12):
        script.append(_write("app.py", f"v{i}\n"))
        script.append(_pytest_run("test_a", "test_b"))  # byte-for-byte the same failure
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), PROMPT)

    assert trace.error is None
    assert _nudge_reached_model(app)
    assert getattr(app, "_last_turn_loop_exit_reason", "") == "plateau_exhausted"


def test_recent_pass_after_nudge_yields_finish_now_instead_of_stop(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    script = [_write("app.py", "v0\n")]                               # round 0
    script.append(_pytest_run())                                      # round 1: first pass = progress
    script += [_write("app.py", f"v{i}\n") for i in range(1, 7)]      # rounds 2..7: 6 no-progress → nudge at 7
    script += [_write("app.py", f"w{i}\n") for i in range(6)]         # rounds 8..13: 6 more no-progress
    script.append(_pytest_run())                                      # round 14: pass-after-pass (no progress, 7th)
    script.append(_write("app.py", "final\n"))                        # round 15: 8th → would stop; pass 1 round ago
    script.append(_write("README.md", "# CSV CLI\n"))                  # round 16: new file (progress)
    script.append(say("Done."))
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), PROMPT)

    assert trace.error is None
    assert _nudge_reached_model(app), "wrap-up nudge should have fired"
    assert _nudge_reached_model(app, FINISH_MARK), "finish-now nudge should have fired"
    assert getattr(app, "_last_turn_loop_exit_reason", "") != "plateau_exhausted"
    finish = _nudge_texts(app, FINISH_MARK)
    assert len(finish) == 1
    assert "finish now: write any remaining deliverable and stop." in finish[0]
    assert len(app.engine.calls) >= len(script)


def test_wrap_up_nudge_names_open_todo_items(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    script = [_plan_round(("Write parser", "in_progress"), ("Create rollback.sql", "pending"), ("Add README", "pending"))]
    script.append(_write("app.py", "v0\n"))
    script += [_write("app.py", f"v{i}\n") for i in range(1, 30)]
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), PROMPT)

    assert trace.error is None
    nudges = _nudge_texts(app)
    assert nudges, "wrap-up nudge never reached the model"
    assert "Open plan items:" in nudges[0]
    assert "- Create rollback.sql" in nudges[0]
    assert "- Add README" in nudges[0]
    assert "- Write parser" in nudges[0]
    # …and the exhausted summary lists them the same way.
    assert getattr(app, "_last_turn_loop_exit_reason", "") == "plateau_exhausted"
    assert "- Create rollback.sql" in (trace.final_response or "")
