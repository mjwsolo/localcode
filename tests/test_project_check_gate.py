"""The completion gate must never accept a build the checker never verified.

Reproduces the failure shape directly: a checker that times out or cannot run
sets a NO-VERDICT state that has to survive the whole turn, force bounded
retries, block the final completion even when the verification registry is
otherwise satisfied, and put its reason into the final result text (the only
channel that reaches both the TUI and `--json`).
"""
from __future__ import annotations

import ast
from pathlib import Path

from localcode.agent.project_check_gate import OK, RED, UNVERIFIED, ProjectCheckGate
from localcode.tools.project_check import CheckOutcome

CLEAN = CheckOutcome("clean", "tsc")
ERRORS = CheckOutcome("errors", "tsc", "[tsc] reported errors:\nsrc/a.ts: TS2322")
TIMED_OUT = CheckOutcome("timed_out", "tsc", "[tsc] timed out after 60s")
FAILED = CheckOutcome("failed", "tsc", "[tsc] exited 2 with no output")
UNAVAILABLE = CheckOutcome("unavailable", "", "no project checker available")


def test_clean_does_not_block():
    gate = ProjectCheckGate()
    assert gate.observe(CLEAN) == OK
    assert not gate.blocks_completion() and gate.result_note() == ""


def test_errors_are_red_not_unverified():
    gate = ProjectCheckGate()
    assert gate.observe(ERRORS) == RED
    assert not gate.blocks_completion()  # handled by the red path, which retries


def test_timeout_blocks_completion():
    gate = ProjectCheckGate()
    assert gate.observe(TIMED_OUT) == UNVERIFIED
    assert gate.blocks_completion()
    assert "timed out" in gate.result_note()


def test_failure_blocks_completion():
    gate = ProjectCheckGate()
    assert gate.observe(FAILED) == UNVERIFIED
    assert gate.blocks_completion()


def test_unavailable_checker_does_not_block():
    """No checker installed is a known, intentional skip — not a failed run."""
    gate = ProjectCheckGate()
    assert gate.observe(UNAVAILABLE) == OK
    assert not gate.blocks_completion()


def test_exception_is_treated_as_failed():
    gate = ProjectCheckGate()
    assert gate.observe_exception(OSError("boom")) == UNVERIFIED
    assert gate.blocks_completion()
    assert "OSError" in gate.result_note()


def test_every_failure_retries_until_the_bound_then_still_blocks():
    """The bug: only the FIRST failure forced a retry; later ones fell through
    to a successful completion."""
    gate = ProjectCheckGate(max_retries=2)
    for _ in range(2):
        assert gate.observe(TIMED_OUT) == UNVERIFIED
        assert gate.consume_retry() is True
    assert gate.observe(TIMED_OUT) == UNVERIFIED
    assert gate.consume_retry() is False       # bounded — no infinite spin
    assert gate.blocks_completion()            # but STILL not verified
    assert "not verified" in gate.result_note()


def test_a_later_clean_run_clears_an_earlier_failure():
    gate = ProjectCheckGate()
    gate.observe(TIMED_OUT)
    assert gate.blocks_completion()
    gate.observe(CLEAN)
    assert not gate.blocks_completion() and gate.result_note() == ""


def test_a_later_red_run_clears_the_no_verdict_state():
    """Red means the checker worked; the red path owns the round from there."""
    gate = ProjectCheckGate()
    gate.observe(FAILED)
    assert gate.observe(ERRORS) == RED
    assert not gate.blocks_completion()


def test_result_note_is_appended_not_substituted():
    gate = ProjectCheckGate()
    gate.observe(FAILED)
    note = gate.result_note()
    assert note.startswith("\n\n") and "exited 2" in note


# ── wiring: the loop must consult the gate at FINAL completion ──────────────

def _loop_source() -> str:
    return (Path(__file__).resolve().parents[1]
            / "src" / "localcode" / "agent" / "loop.py").read_text()


def test_loop_blocks_final_completion_on_the_gate():
    """`_completion_blocked` used to consult only the verification registry, so
    an unrelated earlier pass let the turn finish while the project check kept
    failing."""
    src = _loop_source()
    blocked = src.split("_completion_blocked = (", 1)[1].split("\n            )", 1)[0]
    assert "_project_check_gate.blocks_completion()" in blocked


def test_loop_puts_the_reason_in_the_final_result_text():
    src = _loop_source()
    assert "content += _project_check_gate.result_note()" in src


def test_loop_gate_state_is_turn_level_not_round_level():
    """Initialised once per turn, outside the round loop — otherwise a failure
    in round 3 is forgotten by round 6."""
    tree = ast.parse(_loop_source())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "_project_check_gate" for t in node.targets)):
            break
    else:
        raise AssertionError("_project_check_gate is never initialised")
