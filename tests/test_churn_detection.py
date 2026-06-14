"""Semantic-churn detection — the loop-breaker for thrashing-without-converging.

Covers detect_churn / command_token / churn_nudge_for (pure functions). The
byte-identical-call breakers miss these patterns (different content / output
each time); these guards catch a file rewritten N times, a command failing N
times, and a long read-only investigation spin — without false-positiving on
normal multi-step work.
"""
from __future__ import annotations

from localcode.agent.recovery import (
    ChurnMode,
    detect_churn,
    churn_nudge_for,
    command_token,
)
from localcode.agent.constants import (
    CHURN_FILE_WRITE_LIMIT,
    CHURN_COMMAND_FAIL_LIMIT,
    CHURN_READONLY_STREAK_LIMIT,
    CHURN_PLANNING_STREAK_LIMIT,
)


def test_command_token_normalizes_family():
    assert command_token("npm i") == "npm"
    assert command_token("FOO=bar npm run build") == "npm"   # env prefix skipped
    assert command_token("sudo npm install") == "npm"        # sudo skipped
    assert command_token("nohup node server.js &") == "node"
    assert command_token("   ") == ""


def test_file_rewrite_churn_fires_at_threshold():
    counts = {"package.json": CHURN_FILE_WRITE_LIMIT}
    sig = detect_churn(counts, {}, 0)
    assert sig is not None and sig.mode is ChurnMode.FILE_REWRITE
    assert sig.subject == "package.json"
    assert sig.count == CHURN_FILE_WRITE_LIMIT


def test_command_failure_churn_fires_at_threshold():
    sig = detect_churn({}, {"npm": CHURN_COMMAND_FAIL_LIMIT}, 0)
    assert sig is not None and sig.mode is ChurnMode.COMMAND_FAILURE
    assert sig.subject == "npm"


def test_investigation_spin_fires_at_threshold():
    sig = detect_churn({}, {}, CHURN_READONLY_STREAK_LIMIT)
    assert sig is not None and sig.mode is ChurnMode.INVESTIGATION_SPIN


def test_planning_spin_fires_at_threshold():
    sig = detect_churn({}, {}, 0, planning_streak=CHURN_PLANNING_STREAK_LIMIT)
    assert sig is not None and sig.mode is ChurnMode.PLANNING_SPIN
    assert sig.count == CHURN_PLANNING_STREAK_LIMIT


def test_planning_spin_below_threshold_does_not_fire():
    assert detect_churn({}, {}, 0, planning_streak=CHURN_PLANNING_STREAK_LIMIT - 1) is None


def test_planning_streak_defaults_to_zero_for_legacy_callers():
    # Callers that don't pass planning_streak keep the prior three-signal
    # behaviour — no PLANNING_SPIN without the new input.
    assert detect_churn({}, {}, 0) is None
    assert detect_churn({}, {}, CHURN_READONLY_STREAK_LIMIT - 1) is None


def test_investigation_spin_outranks_planning_spin():
    # Both soft signals tripped → INVESTIGATION_SPIN wins (keeps its distinct
    # "stop reading in circles" framing for the pure read-only case).
    sig = detect_churn(
        {}, {},
        CHURN_READONLY_STREAK_LIMIT,
        planning_streak=CHURN_PLANNING_STREAK_LIMIT + 3,
    )
    assert sig.mode is ChurnMode.INVESTIGATION_SPIN


def test_planning_spin_nudge_is_actionable():
    sig = detect_churn({}, {}, 0, planning_streak=CHURN_PLANNING_STREAK_LIMIT)
    text = churn_nudge_for(sig)
    assert "planned enough" in text.lower()
    assert "concrete action" in text.lower() or "execute" in text.lower()


def test_command_failure_takes_precedence_over_file_rewrite():
    # Both tripped → COMMAND_FAILURE wins (most actionable: a concrete error).
    sig = detect_churn(
        {"a.ts": CHURN_FILE_WRITE_LIMIT + 2},
        {"npm": CHURN_COMMAND_FAIL_LIMIT},
        CHURN_READONLY_STREAK_LIMIT + 5,
    )
    assert sig.mode is ChurnMode.COMMAND_FAILURE


def test_no_false_positive_below_thresholds():
    # A normal flow: wrote a file twice, ran a command that failed once, read
    # a few files. Nothing should trip.
    assert detect_churn(
        {"App.tsx": CHURN_FILE_WRITE_LIMIT - 1},
        {"npm": CHURN_COMMAND_FAIL_LIMIT - 1},
        CHURN_READONLY_STREAK_LIMIT - 1,
    ) is None
    assert detect_churn({}, {}, 0) is None


def test_nudge_text_names_subject_and_is_actionable():
    fr = churn_nudge_for(detect_churn({"db.ts": CHURN_FILE_WRITE_LIMIT}, {}, 0))
    assert "db.ts" in fr and "edit_file" in fr.lower()
    cf = churn_nudge_for(detect_churn({}, {"npm": CHURN_COMMAND_FAIL_LIMIT}, 0))
    assert "npm" in cf and "root cause" in cf.lower()
    spin = churn_nudge_for(detect_churn({}, {}, CHURN_READONLY_STREAK_LIMIT))
    assert "circles" in spin.lower() or "concrete action" in spin.lower()
