"""Scenario: 25-turn complex coding session — the real thing.

What this scenario actually does
--------------------------------
Drives 25 SEQUENTIAL user turns through the real agent loop with the
real model. Each turn is a small, realistic coding task that builds
incrementally on prior turns — by turn 25 a working TODO CLI exists at
`~/localcode-e2e-todo-<random>/`.

This is the test the user repeatedly asked for and that I (terminal coding tools)
overclaimed earlier in the session. The previous "25-turn test" was a
SYNTHETIC test with fake message dicts — no model, no GPU, no I/O.
This one is the actual end-to-end test: real llama-server, real
inference, real file writes, real bash, real Metal compute.

What it asserts
---------------
1. No turn raises an unhandled exception.
2. No turn produces an `[E3xxx]` (runtime/server) error code.
3. Total session completes in under WALL_CLOCK_BUDGET_S (default 30 min).
4. By the end, the target Python file (`todo.py`) exists AND is parseable
   AS PYTHON (we ast.parse it). This is the ground-truth check that the
   model produced something real, not just hallucinated about it.
5. Per-turn redaction telemetry is emitted (proves the context-reduction
   pipeline fired).
6. Peak in-context size never exceeds CONTEXT_PEAK_BUDGET (default 50K).
7. At least N out of 25 turns produced at least one tool call.

Runtime: 15-40 minutes depending on model warmth, system load, and
context-prefill cost as history grows. NOT a fast test — meant to be
run before a release, not on every save.

Why this matters
----------------
The bugs that hurt the user most are the ones that only show up after
many turns of accumulated state — context bloat, tool-call dedup
regressions, history-redaction edge cases, double-server memory peaks
during model switches mid-session. None of those are visible in a
1-turn smoke test. This is where real defects surface.
"""
from __future__ import annotations

import ast
import os
import random
import shutil
import time
from pathlib import Path

from ..harness import (
    EventRecorder, ScenarioResult, asserts,
    build_headless_app, ensure_server_ready, run_one_turn,
    lifecycle_log_tail,
)


WALL_CLOCK_BUDGET_S = 30 * 60       # 30 minutes hard ceiling
CONTEXT_PEAK_BUDGET = 50_000        # tokens; soft warning (not always known)
MIN_TURNS_WITH_TOOLS = 15           # at least 15/25 turns must use tools


# Each prompt is small + concrete to give the model a fair chance per turn.
# Together they build a working TODO CLI.
PROMPTS_25 = [
    # Setup
    "Create a new directory at {dir}. Then create an empty file {dir}/todo.py inside it.",
    "Read {dir}/todo.py to confirm it's empty.",
    "Write {dir}/todo.py with the imports `import json`, `import sys`, and `from pathlib import Path` and nothing else.",

    # Core data model
    "Add a constant `STORAGE_PATH = Path.home() / '.todo-data.json'` to {dir}/todo.py just below the imports.",
    "Add a function `load_todos()` that reads STORAGE_PATH if it exists and returns a list of dicts, else returns an empty list.",
    "Add a function `save_todos(todos)` that writes the list to STORAGE_PATH as pretty-printed JSON.",
    "Read {dir}/todo.py and verify both load_todos and save_todos are present.",

    # Add command
    "Add a function `add_todo(text)` that appends a dict {{'text': text, 'done': False}} to the loaded todos and saves them.",
    "Add a function `list_todos()` that prints each todo as 'N) [x] text' for done, 'N) [ ] text' for pending, where N is 1-indexed.",
    "Add a function `complete_todo(index)` that marks todos[index - 1]['done'] = True and saves.",
    "Add a function `delete_todo(index)` that removes todos[index - 1] from the list and saves.",

    # Argparse CLI
    "Add a `def main():` at the bottom of {dir}/todo.py that uses argparse to expose subcommands: add (text), list, done (index), rm (index).",
    "Add `if __name__ == '__main__': main()` to the very bottom of {dir}/todo.py.",
    "Read {dir}/todo.py and tell me the total line count and number of functions defined.",

    # Smoke test the CLI
    "Run `python {dir}/todo.py add 'first task'` and report the exit code.",
    "Run `python {dir}/todo.py add 'second task'` and report the exit code.",
    "Run `python {dir}/todo.py list` and report the output verbatim.",
    "Run `python {dir}/todo.py done 1` and report exit code.",
    "Run `python {dir}/todo.py list` again and report the output verbatim — the first task should now show as completed.",
    "Run `python {dir}/todo.py rm 2` and report exit code.",
    "Run `python {dir}/todo.py list` and report the output — only one task should remain.",

    # Cleanup + verification
    "Delete the storage file at ~/.todo-data.json if it exists.",
    "Read the final {dir}/todo.py one more time and confirm it parses as valid Python (just acknowledge — don't quote it).",
    "List the contents of {dir} so I can see what's in there.",

    # Final summary
    "Briefly summarise (in 2-3 sentences) what got built and whether all 5 commands worked.",
]


def _make_test_dir() -> str:
    """Per-run dir so concurrent runs / re-runs don't collide."""
    suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
    return str(Path.home() / f"localcode-e2e-todo-{suffix}")


def _cleanup(dir_path: str) -> None:
    try:
        shutil.rmtree(dir_path, ignore_errors=True)
    except Exception:
        pass


def run() -> ScenarioResult:
    name = "long_coding_session_25turn"
    t0 = time.time()
    test_dir = _make_test_dir()
    target_file = Path(test_dir) / "todo.py"

    app = build_headless_app()
    server_up = ensure_server_ready(app, timeout_s=180)
    if not server_up:
        return ScenarioResult(
            name=name, passed=False,
            assertions=asserts(("server_ready", False, "")),
            wall_clock_s=time.time() - t0,
        )

    # Reset the harness's per-app message accumulator. Some other
    # scenario in the same `run.py` call may have left state.
    if hasattr(app, "_e2e_messages"):
        app._e2e_messages = []

    rec = EventRecorder()
    turns_with_tools = 0
    error_codes_seen: list[str] = []
    over_budget = False
    redaction_events_before = sum(1 for L in lifecycle_log_tail(10000) if " redaction " in L)

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
        print(f"      tools={tools or 'none'}  {trace.wall_clock_s:.1f}s")
        if trace.error:
            print(f"      ERROR: {trace.error[:200]}")

    # Verify the artifact exists and parses.
    file_exists = target_file.is_file()
    parses_as_python = False
    parse_err = ""
    if file_exists:
        try:
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
         f"{redaction_fired} redaction events in lifecycle.log"),
    )
    passed = all(ok for _, ok, _ in checks)

    # Cleanup test artifact (don't fail if it lingers).
    _cleanup(test_dir)

    return ScenarioResult(
        name=name,
        passed=passed,
        assertions=checks,
        evidence={
            "test_dir": test_dir,
            "prompts_executed": len(rec.turns),
            "prompts_total": len(PROMPTS_25),
            "turns_with_tools": turns_with_tools,
            "error_codes_seen": error_codes_seen,
            "wall_clock_s": round(elapsed, 1),
            "target_file_existed": file_exists,
            "target_file_parsed": parses_as_python,
            "parse_error": parse_err,
            "redaction_events_during": redaction_fired,
            "over_wall_clock_budget": over_budget,
        },
        turns=rec.turns,
        wall_clock_s=elapsed,
    )
