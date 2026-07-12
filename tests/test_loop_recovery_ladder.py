"""Recovery-ladder pure functions added for the stop-hook / bounded-recovery work.

Covers three new helpers in agent/recovery.py:

  • detect_reject_reread_loop  — the reject → re-read → reject spin, recomputed
    from the transcript tail (survives compaction).
  • reject_reread_nudge        — its one-shot forward-only redirect.
  • todo_close_verification_suffix — the todo-close "verify before you finish"
    reminder that rides on the todo_write tool result.

These mirror claude-code's stop-hook blocking-error injection, its single-shot
recovery guards, and its TodoWriteTool verification nudge — generalized so they
only fire when there is something to recover from.
"""
from __future__ import annotations

from localcode.agent.recovery import (
    REJECT_REREAD_LOOP_LIMIT,
    detect_reject_reread_loop,
    reject_reread_nudge,
    todo_close_verification_suffix,
)


def _reject(content: str = "REJECTED: dedup") -> dict:
    return {"role": "tool", "content": content}


def _call(name: str) -> dict:
    return {"role": "assistant", "tool_calls": [{"function": {"name": name}}]}


# ── detect_reject_reread_loop ────────────────────────────────────────────────

def test_reject_reread_fires_at_threshold_with_reread():
    msgs = [
        _call("edit_file"), _reject("REJECTED: dedup"),
        _call("read_file"), {"role": "tool", "content": "file body"},
        _call("edit_file"), _reject("REJECTED: modified since read"),
        _call("edit_file"), _reject("REJECTED: old_string not found"),
    ]
    assert detect_reject_reread_loop(msgs) == REJECT_REREAD_LOOP_LIMIT


def test_reject_reread_needs_a_reread_tell():
    # Rejections galore but the model never re-reads — that's a plain
    # repeated-failure (handled elsewhere), not this spin.
    msgs = [_reject() for _ in range(REJECT_REREAD_LOOP_LIMIT + 1)]
    assert detect_reject_reread_loop(msgs) is None


def test_reject_reread_below_threshold_is_none():
    msgs = [
        _call("edit_file"), _reject(),
        _call("read_file"), {"role": "tool", "content": "body"},
    ]
    assert detect_reject_reread_loop(msgs) is None


def test_reject_reread_ignores_non_rejection_tool_results():
    msgs = [
        _call("read_file"), {"role": "tool", "content": "ok, file body"},
        _call("edit_file"), {"role": "tool", "content": "Wrote 10 lines."},
    ]
    assert detect_reject_reread_loop(msgs) is None


def test_reject_reread_only_counts_the_window_tail():
    # Old rejections outside the window must not count — the signal is
    # recomputed from the recent transcript so it decays as the model recovers.
    old = [_reject() for _ in range(5)]
    recent_clean = [
        _call("read_file"), {"role": "tool", "content": "body"},
        _call("edit_file"), {"role": "tool", "content": "Wrote file."},
    ]
    assert detect_reject_reread_loop(old + recent_clean, window=4) is None


def test_reject_reread_nudge_is_forward_only():
    text = reject_reread_nudge()
    assert text.startswith("SYSTEM:")
    lowered = text.lower()
    # forward-only imperative + escape hatch, no echo-able self-critical loop talk
    assert "edit_file" in lowered
    assert "move on" in lowered
    for banned in ("going in circles", "you keep failing", "thrashing"):
        assert banned not in lowered


# ── todo_close_verification_suffix ───────────────────────────────────────────

def _todo(content: str, status: str = "completed") -> dict:
    return {"content": content, "status": status}


def test_todo_suffix_fires_on_3plus_all_done_no_verify():
    todos = [_todo("add login"), _todo("add signup"), _todo("style the page")]
    suffix = todo_close_verification_suffix(todos)
    assert suffix and "verification step" in suffix
    assert suffix.startswith("\n\n")  # rides on an existing tool result


def test_todo_suffix_silent_when_an_item_mentions_verification():
    for verify_word in ("write tests", "run the typecheck", "lint the code",
                        "build the project", "verify output"):
        todos = [_todo("add login"), _todo(verify_word), _todo("ship")]
        assert todo_close_verification_suffix(todos) == ""


def test_todo_suffix_silent_below_three_items():
    assert todo_close_verification_suffix([_todo("a"), _todo("b")]) == ""


def test_todo_suffix_silent_when_not_all_completed():
    todos = [_todo("a"), _todo("b", status="in_progress"), _todo("c")]
    assert todo_close_verification_suffix(todos) == ""


def test_todo_suffix_handles_junk_input():
    assert todo_close_verification_suffix(None) == ""
    assert todo_close_verification_suffix([]) == ""
    assert todo_close_verification_suffix(["not a dict", 3]) == ""
