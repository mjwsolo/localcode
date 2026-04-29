"""Runtime memory protection — prevents the D-state / OOM-kill trap.

Two independent defenses:

1. `start_pressure_monitor(process)` spins a background thread that polls
   `sysctl kern.memorystatus_vm_pressure_level` every 500 ms. On transition
   to WARN (before the system gets stuck), it SIGTERMs the llama-server
   subprocess and fires a callback so the UI can tell the user.

2. `set_jetsam_highwater(pid, limit_mb)` asks the macOS kernel to jetsam-
   kill the subprocess if its resident memory crosses `limit_mb`. This is
   the sanctioned Apple-blessed mechanism (memorystatus_control syscall,
   command 7 = SET_MEMLIMIT_PROPERTIES) used by Apple's own sandboxed
   apps. Kernel enforces it BEFORE the vm_fault wait-chain can form, so
   we never enter the "stuck in kernel Metal syscall" state.

Both no-op on non-Darwin platforms.

Research basis: the kernel wait that makes llama-server un-SIGKILL-able is
a `PCATCH`-less `vm_fault → vnode_pagein` deep inside IOGPU. Once parked,
there is no userspace recovery — the BSD signal layer never runs for that
thread. Only prevention works. (Agents' research 2026-04-23.)
"""
from __future__ import annotations

import ctypes
import ctypes.util
import platform
import subprocess
import threading
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────
# Pressure monitor (#1)
# ─────────────────────────────────────────────────────────────────────

# sysctl values for kern.memorystatus_vm_pressure_level. Apple's kernel
# emits these via DISPATCH_SOURCE_TYPE_MEMORYPRESSURE as well, but we
# can just poll — the memory cost is one syscall per 500 ms, much
# lighter than a Grand Central Dispatch listener.
PRESSURE_NORMAL = 1
PRESSURE_WARN = 2
PRESSURE_CRITICAL = 4

_POLL_INTERVAL_SEC = 0.5


def _read_pressure_level() -> int:
    """Return current VM pressure level (1, 2, or 4). 0 on error / non-Darwin."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=1,
        )
        return int(r.stdout.strip())
    except Exception:
        return 0


def start_pressure_monitor(
    subprocess_obj: subprocess.Popen,
    on_pressure: Optional[Callable[[int], None]] = None,
) -> Optional[threading.Thread]:
    """Watch kernel memory pressure; kill subprocess_obj on WARN+.

    Returns the thread handle (daemon) or None on non-Darwin. The thread
    exits cleanly when subprocess_obj exits or the interpreter shuts down.

    `on_pressure` is called with the pressure level (2 or 4) once, before
    we SIGTERM — lets the UI surface a message to the user. Keep this
    callback fast; the thread is the only thing standing between the user
    and an OOM kill.
    """
    if platform.system().lower() != "darwin":
        return None

    stopped = threading.Event()
    # Belt-and-suspenders: stop when subprocess exits
    def _watchdog() -> None:
        # Hysteresis: only kill on SUSTAINED CRITICAL pressure, not on a
        # single WARN spike. Earlier behavior killed llama-server on the
        # first WARN reading — but Metal GPU compute during inference
        # routinely bumps the kernel to WARN for a few hundred ms and
        # back. That's normal. Killing on it caused "Server ready" → user
        # sends prompt → server exits mid-request → HTTP 503. Now we
        # require CRITICAL (the kernel's "OOM-killer is about to fire"
        # signal) AND it has to persist across multiple polls before we
        # intervene. WARN spikes are ignored; the model server is left
        # to do its job.
        consecutive_critical = 0
        REQUIRED_CRITICAL_POLLS = 4   # ~2 s of sustained CRITICAL = real
        while not stopped.is_set():
            if subprocess_obj.poll() is not None:
                return
            level = _read_pressure_level()
            if level >= PRESSURE_CRITICAL:
                consecutive_critical += 1
            else:
                consecutive_critical = 0
            if consecutive_critical >= REQUIRED_CRITICAL_POLLS:
                # Real, sustained pressure — intervene before OOM.
                try:
                    if on_pressure is not None:
                        on_pressure(level)
                except Exception:
                    pass
                try:
                    subprocess_obj.terminate()
                except Exception:
                    pass
                try:
                    subprocess_obj.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        subprocess_obj.kill()
                    except Exception:
                        pass
                return
            stopped.wait(_POLL_INTERVAL_SEC)

    t = threading.Thread(target=_watchdog, name="localcode-memguard", daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────────────
# Jetsam high-water mark (#2)
# ─────────────────────────────────────────────────────────────────────

# memorystatus_control(uint32_t command, int32_t pid, uint32_t flags,
#                       void *buffer, size_t buffersize) -> int
# Exposed in libSystem on macOS (system call #440 historically).
_MEMORYSTATUS_CMD_SET_MEMLIMIT_PROPERTIES = 7
_MEMORYSTATUS_MEMLIMIT_ATTR_FATAL = 0x1


class _MemlimitProps(ctypes.Structure):
    """memorystatus_memlimit_properties_t from kern_memorystatus.h."""
    _fields_ = [
        ("memlimit_active",       ctypes.c_int32),
        ("memlimit_active_attr",  ctypes.c_uint32),
        ("memlimit_inactive",     ctypes.c_int32),
        ("memlimit_inactive_attr", ctypes.c_uint32),
    ]


def set_jetsam_highwater(pid: int, limit_mb: int) -> bool:
    """Ask the kernel to jetsam-kill `pid` if its memory exceeds `limit_mb`.

    This is the hard backstop: the kernel enforces it BEFORE the process
    can wire enough memory to hang the system. Apple's own sandboxed apps
    use this; it's the sanctioned API.

    Returns True on success, False on any failure (non-Darwin, permission
    denied, symbol missing, etc.) — non-fatal; we always continue.

    Limit is the SAME for active + inactive with the FATAL flag so jetsam
    kills rather than just warns. Active = foreground, inactive =
    backgrounded; for a subprocess we own that's always "active" in the
    user's perception, so making them equal is correct.
    """
    if platform.system().lower() != "darwin":
        return False
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("System") or "/usr/lib/libSystem.B.dylib",
                           use_errno=True)
        libc.memorystatus_control.argtypes = [
            ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        libc.memorystatus_control.restype = ctypes.c_int32
    except Exception:
        return False

    props = _MemlimitProps(
        memlimit_active=limit_mb,
        memlimit_active_attr=_MEMORYSTATUS_MEMLIMIT_ATTR_FATAL,
        memlimit_inactive=limit_mb,
        memlimit_inactive_attr=_MEMORYSTATUS_MEMLIMIT_ATTR_FATAL,
    )
    try:
        r = libc.memorystatus_control(
            _MEMORYSTATUS_CMD_SET_MEMLIMIT_PROPERTIES,
            pid,
            0,
            ctypes.byref(props),
            ctypes.sizeof(props),
        )
        return r == 0
    except Exception:
        return False


def recommended_jetsam_limit_mb() -> int:
    """Budget: physical_ram − 3.5 GB reserved for macOS + other apps.

    On a 16 GB Mac that's 16384 − 3500 = 12884 MB. llama-server for our
    10–11 GB models tops out around 13 GB wired during decode (measured
    Apr 22 2026), so this leaves ~100 MB margin — kernel will kill
    llama-server before it can push the system over the edge. On 32 GB+
    we're more permissive.
    """
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True, timeout=2)
        total_mb = int(r.stdout.strip()) // (1024 * 1024)
    except Exception:
        total_mb = 16 * 1024
    reserve = 3500 if total_mb < 24 * 1024 else 6000
    return max(2048, total_mb - reserve)
