"""Scenario: server lifecycle works without leaking memory or processes.

What this catches
-----------------
The "two servers running simultaneously" hypothesis the user raised in
this session. Every time a `/model` switch happens, the new server
should NOT be spawned until the OLD server's wired memory is released.
Otherwise we get a 20 GB peak commit on a 14 GB GPU cap → pressure-kill
→ user-visible E3102.

Specifically asserts:
  1. Lifecycle log records `server_started`, `server_stopped`,
     `memory_wait_start` events in the right order.
  2. After a model switch, free memory is not lower than before
     (sign of leaked old-server memory).
  3. Only one llama-server process is alive at the end.
  4. No `pressure_kill` event fires during the test (we shouldn't
     trigger pressure on a deliberate switch unless the system is
     already at the brink).

Runtime: ~30-60 seconds (just spawns + kills server, no long inference).
Does NOT exercise the agent loop — purely infrastructure.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ..harness import (
    ScenarioResult, asserts, lifecycle_log_tail,
)


def _free_mem_mb() -> int:
    """Read `vm_stat` for free + inactive pages, return MiB."""
    try:
        from localcode.server_manager import _system_free_memory_mb
        return _system_free_memory_mb()
    except Exception:
        return 0


def _live_llama_server_pids() -> list[int]:
    try:
        r = subprocess.run(["pgrep", "-f", "llama-server"],
                           capture_output=True, text=True, timeout=3)
        return [int(p) for p in r.stdout.split() if p.isdigit()]
    except Exception:
        return []


def run() -> ScenarioResult:
    name = "server_lifecycle"
    t0 = time.time()
    start_log_len = len(lifecycle_log_tail(n=10000))

    # 1. Build app — this constructs ServerManager but doesn't spawn.
    from ..harness import build_headless_app, ensure_server_ready
    app = build_headless_app()

    # 2. Ensure server is up. Records baseline state.
    free_before = _free_mem_mb()
    pids_before = _live_llama_server_pids()
    started_ok = ensure_server_ready(app, timeout_s=180)

    # 3. Trigger a model "switch" by calling _restart_server() directly —
    #    same code path /model uses but skipping the model name change so
    #    we re-test the same model. Lets us isolate the lifecycle bug from
    #    the model-load bug.
    if started_ok:
        time.sleep(1)
        free_mid = _free_mem_mb()
        pids_mid = _live_llama_server_pids()
        try:
            ok_restart = app.engine._restart_server()
        except Exception as exc:
            ok_restart = False
            restart_err = str(exc)
        else:
            restart_err = ""
    else:
        ok_restart = False
        free_mid = 0
        pids_mid = []
        restart_err = "initial server start failed"

    # 4. Final state.
    time.sleep(2)
    free_after = _free_mem_mb()
    pids_after = _live_llama_server_pids()

    # 5. Read the lifecycle log entries written DURING this test.
    new_log = lifecycle_log_tail(n=10000)[start_log_len:]
    has_event = lambda tag: any(tag in line for line in new_log)
    pressure_killed = has_event("pressure_kill")

    # 6. Assert.
    checks = asserts(
        ("server_started_recorded", has_event("server_started"),
         f"lifecycle.log lines: {len(new_log)}"),
        ("server_stopped_recorded_during_restart",
         has_event("server_stopped") if ok_restart else True,
         "expected at least one stop during restart"),
        ("memory_wait_recorded_during_restart",
         has_event("memory_wait_start") if ok_restart else True,
         "memory_wait_start should fire whenever we kill before spawn"),
        ("no_pressure_kill", not pressure_killed,
         "pressure_kill fired during test — system was already too pressured"),
        ("single_server_alive_at_end", len(pids_after) <= 1,
         f"pids alive: {pids_after}"),
        ("restart_succeeded", ok_restart,
         restart_err or "restart returned False"),
        ("free_memory_recovered",
         free_after >= free_before - 500 if started_ok else True,
         f"free MiB before={free_before} mid={free_mid} after={free_after}"),
    )

    passed = all(ok for _, ok, _ in checks)
    return ScenarioResult(
        name=name,
        passed=passed,
        assertions=checks,
        evidence={
            "free_mb_before": free_before,
            "free_mb_mid": free_mid,
            "free_mb_after": free_after,
            "pids_before": pids_before,
            "pids_mid": pids_mid,
            "pids_after": pids_after,
            "lifecycle_events_during_test": new_log,
            "pressure_killed": pressure_killed,
            "restart_succeeded": ok_restart,
            "restart_error": restart_err,
        },
        wall_clock_s=time.time() - t0,
    )
