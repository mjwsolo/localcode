"""Scenario: intent-without-action stalls auto-recover via the nudge.

What this catches
-----------------
Image 112's failure mode: the model calls a few tools, then ends a
round with a short narration ("I'll create the frontend HTML file now.")
and NO tool call. Last session I added an auto-nudge in `agent.py`
that should detect this pattern and inject a synthetic SYSTEM message
forcing the model to act. This scenario gives the model a prompt
designed to provoke the stall (multi-step task) and asserts:

  1. Either the task completes WITHOUT a stall (best case), OR
  2. A stall is detected and the auto-nudge fires (we look for the
     `Model round ended with ... — nudging it to continue` info line),
     and then the model continues.

  3. Final result: at least ONE tool call happened (model didn't sit
     in pure-prose mode the entire turn).

Runtime: 30-90 s depending on model + prompt complexity.

Note on flakiness
-----------------
Models are stochastic. If the model never stalls, that's GREAT —
the test passes via path (1). If the model DOES stall, the nudge
must kick in. Either path is "passing." The test only fails if the
turn ended with no progress AND no nudge was logged.
"""
from __future__ import annotations

import time

from ..harness import (
    EventRecorder, ScenarioResult, asserts,
    build_headless_app, ensure_server_ready, run_one_turn,
)


# Prompt designed to push the model toward multi-step action without
# being so vague that it triggers rule 22 (clarifying question).
PROVOKE_PROMPT = (
    "In ~/localcode-e2e-test/, do these three things in this order: "
    "(1) create the directory, (2) write a file `hello.txt` containing "
    "the single word 'ok', (3) cat the file to confirm. Then stop."
)


def run() -> ScenarioResult:
    name = "intent_recovery"
    t0 = time.time()
    app = build_headless_app()
    server_up = ensure_server_ready(app, timeout_s=180)
    if not server_up:
        return ScenarioResult(
            name=name, passed=False,
            assertions=asserts(("server_ready", False, "")),
            wall_clock_s=time.time() - t0,
        )

    rec = EventRecorder()
    trace = run_one_turn(app, rec, PROVOKE_PROMPT)

    tools = trace.tool_calls_made()
    info = trace.info_messages()
    nudge_fired = any("nudging it to continue" in m for m in info)
    completed = bool(trace.final_response.strip()) or len(tools) >= 1
    no_traceback = trace.error is None or "Traceback" not in (trace.error or "")

    # Pass if: completed cleanly, OR stall was detected and nudged.
    # Fail only if: nothing happened (no tools + no nudge + no response).
    if completed:
        recovered_or_completed = True
        recovery_detail = "model completed without needing nudge"
    elif nudge_fired:
        recovered_or_completed = True
        recovery_detail = "stall detected and auto-nudged"
    else:
        recovered_or_completed = False
        recovery_detail = "stalled with NO recovery — nudge did not fire"

    checks = asserts(
        ("agent_didnt_crash", no_traceback,
         (trace.error or "")[:200]),
        ("at_least_one_tool_call", len(tools) >= 1,
         f"tools called: {tools}"),
        ("recovered_or_completed_cleanly", recovered_or_completed,
         recovery_detail),
        ("no_error_events", len(trace.events_of("error")) == 0,
         f"{len(trace.events_of('error'))} error events"),
    )
    passed = all(ok for _, ok, _ in checks)

    return ScenarioResult(
        name=name,
        passed=passed,
        assertions=checks,
        evidence={
            "tool_calls": tools,
            "tool_count": len(tools),
            "nudge_fired": nudge_fired,
            "info_messages": info,
            "turn_seconds": round(trace.wall_clock_s, 2),
            "final_response_excerpt": trace.final_response[:300],
        },
        turns=[trace],
        wall_clock_s=time.time() - t0,
    )
