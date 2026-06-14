from __future__ import annotations

from optimize.candidate import Candidate
from optimize.evaluator import render_feedback
from optimize.reflector import (
    REQUIRED_PLACEHOLDERS,
    build_reflection_prompt,
    extract_proposed_prompt,
)

SEED = "system prompt " + " ".join(REQUIRED_PLACEHOLDERS)


def test_render_feedback_no_failures():
    out = render_feedback([], {"pass_rate": 1.0})
    assert "No failing tasks" in out
    assert "pass_rate=100.0%" in out


def test_render_feedback_includes_task_and_error_and_truncates():
    fail = {
        "id": "decode_string",
        "prompt": "Write decode_string",
        "code": "x" * 2000,
        "error": "AssertionError: nested case",
    }
    out = render_feedback([fail], {"pass_rate": 0.5})
    assert "decode_string" in out
    assert "AssertionError" in out
    # code is truncated well under its 2000 chars
    assert "x" * 700 not in out


def test_build_reflection_prompt_contains_feedback_and_placeholder_rule():
    parent = Candidate(prompt=SEED, score=0.4, feedback="task foo failed: boom")
    meta = build_reflection_prompt(parent)
    assert "task foo failed: boom" in meta
    assert "{cwd}" in meta  # placeholder list is shown to the model
    assert "<NEW_SYSTEM_PROMPT>" in meta


def test_extract_fenced_prompt_ok():
    new = SEED + " IMPROVED"
    raw = f"sure!\n<NEW_SYSTEM_PROMPT>\n{new}\n</NEW_SYSTEM_PROMPT>\ndone"
    assert extract_proposed_prompt(raw, fallback=SEED) == new


def test_extract_falls_back_when_no_fence():
    # No fence and the bare text lacks placeholders -> use fallback (parent).
    assert extract_proposed_prompt("here is a prompt", fallback=SEED) == SEED


def test_extract_falls_back_when_placeholder_dropped():
    broken = "<NEW_SYSTEM_PROMPT>missing the slots</NEW_SYSTEM_PROMPT>"
    assert extract_proposed_prompt(broken, fallback=SEED) == SEED
