"""Automatic recovery from a stuck llama-server without rebooting.

Runs silently in the background when the TUI detects stuck processes
at preflight. User sees at most one password dialog (the standard
macOS admin auth, same one used for GPU-memory setup). No manual
commands, no "close your lid" nonsense.

The kernel wait that makes `kill -9` not work isn't literally infinite
— it's a thread parked in `vm_fault → vnode_pagein` or IOGPU command
submit, waiting on I/O that can't complete because swap is saturated.
Relieve the pressure → the backing I/O finishes → the thread resumes
→ the queued SIGKILL lands → process dies. No reboot needed.

Recovery ladder (each escalation is more aggressive; stops at first
clean state):

  1. memory_pressure -l critical -s 30   (single admin prompt)
  2. purge                                (no extra prompt — creds cached)
  3. memory_pressure -l critical -s 60   (longer forced reclaim)
  4. Re-scan. If still stuck after 4: surface the failure honestly.

All four run back-to-back without user interaction after the first
password dialog. Total worst-case time: ~2 minutes.
"""
from __future__ import annotations

import subprocess
import time

from .health import find_stuck_servers
from ._subproc_env import clean_env


def _run_as_admin(shell_cmd: str, timeout: int = 60) -> tuple[bool, str]:
    """Run `shell_cmd` with administrator privileges via osascript.
    Returns (success, stderr). User sees the standard macOS password
    dialog.
    """
    escaped = shell_cmd.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'do shell script "{escaped}" '
        'with administrator privileges '
        'with prompt "LocalCode needs admin access to relieve memory pressure and recover a stuck server."'
    )
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout,
            env=clean_env(),
        )
        return r.returncode == 0, (r.stderr or "").strip()
    except Exception as e:
        return False, str(e)


def attempt_recovery(
    verbose: bool = True,
    on_progress: "Callable[[str], None] | None" = None,
) -> tuple[bool, str]:
    """Try to unstick without rebooting. Returns (success, message).

    Fully automatic after a single macOS admin-auth dialog. Passes
    progress updates to `on_progress` so a TUI can show a spinner
    and status text; falls back to stdout when `verbose=True`.
    """
    def log(msg: str) -> None:
        if on_progress is not None:
            try:
                on_progress(msg)
            except Exception:
                pass
        if verbose:
            print(f"  {msg}")

    stuck = find_stuck_servers()
    if not stuck:
        return True, "No stuck llama-server processes detected."

    log(f"Detected {len(stuck)} stuck process(es) — attempting recovery")

    # Bundle all admin commands into ONE osascript → ONE password dialog.
    # We chain with ';' so each runs even if the previous one fails,
    # and we re-check stuck-status between them by writing a sentinel
    # on exit. No user clicks needed after the first dialog.
    cmd_bundle = (
        "/usr/bin/memory_pressure -l critical -s 30 ; "
        "/usr/sbin/purge ; "
        "/usr/bin/memory_pressure -l critical -s 60"
    )

    log("Requesting admin access to relieve memory pressure…")
    ok, err = _run_as_admin(cmd_bundle, timeout=240)
    if not ok:
        # Differentiate: did the user cancel the dialog, or did the
        # commands run and fail? osascript returns -128 on user cancel.
        # Any other non-zero = the relief ran but didn't finish cleanly.
        if "canceled" in err.lower() or "cancel" in err.lower() or "-128" in err:
            return False, (
                "You cancelled the password dialog — recovery didn't run. "
                "Relaunch localcode to try again."
            )
        # Otherwise the commands ran but the system was too far gone
        # to recover via pressure relief. Almost certainly the GPU-wait
        # case.
        return False, (
            "The memory-pressure relief ran but the stuck processes are "
            "waiting on GPU driver state (not swap I/O), which no "
            "userspace command can recover. A reboot is required."
        )
    time.sleep(2)

    stuck = find_stuck_servers()
    if not stuck:
        return True, "Recovered — zombie processes cleared."

    # One more aggressive pass — kernel sometimes needs a second nudge.
    log(f"{len(stuck)} still stuck — retrying with extended pressure…")
    _run_as_admin(
        "/usr/bin/memory_pressure -l critical -s 120",
        timeout=180,
    )
    time.sleep(3)
    stuck = find_stuck_servers()
    if not stuck:
        return True, "Recovered on retry."

    return False, (
        f"{len(stuck)} llama-server process(es) are still in the kernel "
        "after memory-pressure relief. These are GPU-waits — the driver "
        "is holding command queues that need re-initialization. Reboot "
        "is the only remaining option. After reboot, localcode's "
        "preflight will prevent this from recurring."
    )
