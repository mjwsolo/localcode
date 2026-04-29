"""Scenario: short chat completes cleanly with no errors.

The simplest possible smoke test that exercises the FULL stack:
real llama-server, real model, real tool dispatch, real OutputManager,
real agent loop. If this fails, nothing else can pass.

What it asserts
---------------
1. Server is healthy at start.
2. A trivial "hello" prompt produces a non-empty response.
3. Wall clock < 60 s (slow path, but model must produce SOMETHING).
4. No `error` events emitted.
5. No `[E3xxx]` or `[E5xxx]` codes in info messages.
6. The full stack didn't raise.

Runtime: 5-30 s depending on warm/cold cache.
"""
from __future__ import annotations

import time

from ..harness import (
    EventRecorder, ScenarioResult, asserts,
    build_headless_app, ensure_server_ready, run_one_turn,
)


def run() -> ScenarioResult:
    name = "short_chat"
    t0 = time.time()

    app = build_headless_app()
    server_up = ensure_server_ready(app, timeout_s=180)
    if not server_up:
        return ScenarioResult(
            name=name, passed=False,
            assertions=asserts(("server_ready", False, "ensure_server_ready returned False")),
            wall_clock_s=time.time() - t0,
        )

    rec = EventRecorder()
    trace = run_one_turn(app, rec, "Reply with the single word: ready")

    # Build assertions from the trace.
    response = trace.final_response or trace.content_text()
    has_response = bool(response.strip())
    error_event_count = len(trace.events_of("error"))
    bad_codes = [m for m in trace.info_messages()
                 if any(c in m for c in ("[E3", "[E5", "[E9"))]
    no_traceback = trace.error is None or "Traceback" not in (trace.error or "")
    fast_enough = trace.wall_clock_s < 60

    checks = asserts(
        ("server_was_ready", server_up, ""),
        ("got_a_response", has_response, f"response chars: {len(response)}"),
        ("no_error_events", error_event_count == 0,
         f"{error_event_count} error events"),
        ("no_error_codes_in_info", not bad_codes,
         f"bad info messages: {bad_codes[:3]}"),
        ("no_python_traceback", no_traceback,
         (trace.error or "")[:200]),
        ("turn_under_60s", fast_enough, f"{trace.wall_clock_s:.1f}s"),
    )
    passed = all(ok for _, ok, _ in checks)

    return ScenarioResult(
        name=name,
        passed=passed,
        assertions=checks,
        evidence={
            "response_excerpt": response[:200],
            "tool_calls_made": trace.tool_calls_made(),
            "turn_seconds": round(trace.wall_clock_s, 2),
            "event_count": len(trace.events),
        },
        turns=[trace],
        wall_clock_s=time.time() - t0,
    )
