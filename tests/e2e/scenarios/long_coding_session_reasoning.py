"""Scenario: 25-turn complex coding session — REASONING MODE.

Same prompt sequence and assertions as `long_coding_session`, but
forces the runtime into reasoning mode (`*-think`) before driving any
turns. The reasoning-mode failure modes are different from fast-mode:
  • runaway thinking (rule 23 cap)
  • intent-without-action stalls after long reasoning
  • model "stops mid-way" while still streaming reasoning
  • thinking-stream repetition

If both fast and reasoning mode pass this scenario, the agent loop is
genuinely robust across modes. If one passes and the other doesn't,
the failures.md tells us exactly which mode regressed and which
turns failed.

Runtime: 20-45 minutes (reasoning mode is slower per turn — ~2-4×
fast mode wall time at the same context size).
"""
from __future__ import annotations

import time

from ..harness import (
    EventRecorder, ScenarioResult, asserts,
    build_headless_app, ensure_server_ready, run_one_turn,
    lifecycle_log_tail,
)
from .long_coding_session import (
    PROMPTS_25, _make_test_dir, _cleanup,
    WALL_CLOCK_BUDGET_S, MIN_TURNS_WITH_TOOLS,
)


def _force_reasoning_mode(app) -> str:
    """Switch the runtime to its `-think` variant for this scenario.

    Returns the prior mode so we can be polite and restore it after.
    Reads `app.config.runtime.laptop_26b_runtime_mode` and toggles to
    the `*-think` form when it isn't already on it. Doesn't restart
    the server here — the runtime reads the mode flag per-stream via
    `use_thinking = ...endswith('-think')`, so the change takes
    effect on the next stream_chat_events() call.
    """
    rt = app.config.runtime
    prior = rt.laptop_26b_runtime_mode
    if not prior.endswith("-think"):
        # Common modes: "speed" / "speed-think", "turbo" / "turbo-think".
        # If the current mode has no `-think` sibling, fall back to
        # "turbo-think" (the documented reasoning mode for 26B).
        if "-" in prior:
            base = prior.rsplit("-", 1)[0]
            rt.laptop_26b_runtime_mode = f"{base}-think"
        else:
            rt.laptop_26b_runtime_mode = f"{prior}-think"
    return prior


def run() -> ScenarioResult:
    name = "long_coding_session_25turn_reasoning"
    t0 = time.time()
    test_dir = _make_test_dir()
    from pathlib import Path
    target_file = Path(test_dir) / "todo.py"

    app = build_headless_app()
    server_up = ensure_server_ready(app, timeout_s=180)
    if not server_up:
        return ScenarioResult(
            name=name, passed=False,
            assertions=asserts(("server_ready", False, "")),
            wall_clock_s=time.time() - t0,
        )

    prior_mode = _force_reasoning_mode(app)
    print(f"  mode switched: {prior_mode} → {app.config.runtime.laptop_26b_runtime_mode}")

    # Reset per-app harness message buffer.
    if hasattr(app, "_e2e_messages"):
        app._e2e_messages = []

    rec = EventRecorder()
    turns_with_tools = 0
    error_codes_seen: list[str] = []
    over_budget = False
    redaction_events_before = sum(1 for L in lifecycle_log_tail(10000) if " redaction " in L)
    thinking_chars_total = 0
    thinking_aborts = 0
    nudges_fired = 0

    print(f"\n  test directory: {test_dir}")
    print(f"  prompts: {len(PROMPTS_25)}\n")

    for i, prompt_template in enumerate(PROMPTS_25, 1):
        prompt = prompt_template.format(dir=test_dir)
        elapsed = time.time() - t0
        if elapsed > WALL_CLOCK_BUDGET_S:
            print(f"  [{i}/{len(PROMPTS_25)}] BUDGET EXCEEDED ({elapsed:.0f}s)")
            over_budget = True
            break
        print(f"  [{i:2d}/{len(PROMPTS_25)}] {prompt[:80]} …")
        try:
            trace = run_one_turn(app, rec, prompt)
        except Exception as exc:
            print(f"      RAISED: {type(exc).__name__}: {exc}")
            continue
        tools = trace.tool_calls_made()
        info = trace.info_messages()
        if tools:
            turns_with_tools += 1
        for m in info:
            if "[E3" in m or "[E5" in m or "[E9" in m:
                error_codes_seen.append(m.split("]", 1)[0] + "]")
            if "exceeded the per-round cap" in m:
                thinking_aborts += 1
            if "nudging it to continue" in m:
                nudges_fired += 1
        thinking_chars_total += len(trace.thinking_text())
        print(f"      tools={tools or 'none'}  think_chars={len(trace.thinking_text())}  "
              f"nudges_far={nudges_fired}  {trace.wall_clock_s:.1f}s")

    # Restore prior mode.
    app.config.runtime.laptop_26b_runtime_mode = prior_mode

    file_exists = target_file.is_file()
    parses_as_python = False
    parse_err = ""
    if file_exists:
        try:
            import ast
            ast.parse(target_file.read_text())
            parses_as_python = True
        except SyntaxError as exc:
            parse_err = f"line {exc.lineno}: {exc.msg}"

    redaction_events_after = sum(1 for L in lifecycle_log_tail(10000) if " redaction " in L)
    redaction_fired = redaction_events_after - redaction_events_before
    elapsed = time.time() - t0

    checks = asserts(
        ("server_ready", server_up, ""),
        ("did_not_blow_budget", not over_budget, f"{elapsed:.0f}s of {WALL_CLOCK_BUDGET_S}s"),
        ("target_file_exists", file_exists, str(target_file)),
        ("target_file_is_valid_python", parses_as_python, parse_err),
        ("no_runtime_errors", not error_codes_seen, f"saw: {error_codes_seen[:5]}"),
        ("min_tool_use", turns_with_tools >= MIN_TURNS_WITH_TOOLS,
         f"{turns_with_tools}/{len(PROMPTS_25)} turns used tools (min {MIN_TURNS_WITH_TOOLS})"),
        ("redaction_pipeline_fired", redaction_fired > 0,
         f"{redaction_fired} redaction events"),
        ("not_too_many_thinking_aborts", thinking_aborts <= 3,
         f"{thinking_aborts} thinking-abort events (cap should rarely fire)"),
    )
    passed = all(ok for _, ok, _ in checks)

    _cleanup(test_dir)

    return ScenarioResult(
        name=name,
        passed=passed,
        assertions=checks,
        evidence={
            "test_dir": test_dir,
            "mode_used": app.config.runtime.laptop_26b_runtime_mode,
            "prompts_executed": len(rec.turns),
            "prompts_total": len(PROMPTS_25),
            "turns_with_tools": turns_with_tools,
            "error_codes_seen": error_codes_seen,
            "thinking_chars_total": thinking_chars_total,
            "thinking_chars_per_turn_avg": (thinking_chars_total // max(1, len(rec.turns))),
            "thinking_aborts": thinking_aborts,
            "nudges_fired": nudges_fired,
            "wall_clock_s": round(elapsed, 1),
            "target_file_existed": file_exists,
            "target_file_parsed": parses_as_python,
            "redaction_events_during": redaction_fired,
            "over_wall_clock_budget": over_budget,
        },
        turns=rec.turns,
        wall_clock_s=elapsed,
    )
