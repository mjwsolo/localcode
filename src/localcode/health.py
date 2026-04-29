"""System health checks for startup diagnostics.

We learned the hard way that once macOS enters a memory-pressure
cascade (swap saturated, wired memory pegged at the iogpu limit, a
llama-server from a prior crashed session stuck in D-state kernel
sleep holding GPU memory), every subsequent localcode launch gets
OOM-killed before it can even render its first screen.

The only real recovery is a reboot. This module exists so we DETECT
that situation at startup and tell the user to reboot — instead of
crashing them into a cryptic `zsh: killed` / `1;19M35;42;18M` spew.

Two entry points:

  * `check_system_health()` — a single preflight call that returns a
    SystemHealth dataclass. The TUI may recover from stuck processes,
    but transient memory pressure is advisory; startup must not become
    a lottery based on macOS's momentary page accounting.

  * `find_stuck_servers()` — the D-state orphan detector used by
    the above. Exposed for CLI tools / diagnostics too.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from ._subproc_env import clean_env


# Thresholds. On macOS, several intuitive signals are unreliable:
#
# * `Pages free` runs near zero on a healthy machine (kernel keeps
#   memory in inactive / purgeable / compressor instead).
# * `vm.swapusage` reports fullness of the swap file, not whether
#   the kernel is actively swapping. macOS keeps cold pages in swap
#   long after pressure drops, so 80 % swap-used is routine.
# * `ps rss` for a process counts mmap'd file-backed pages as
#   resident even when they're evictable. A llama-server with a
#   10 GB mmap'd model can show 11 GB RSS while only 2 GB is wired.
#
# What ACTUALLY matters for our launch:
# We run TurboQuant llama-server with `-ngl 999 --mmap`. The 10 GB
# GGUF is mmap'd from SSD — pages fault in lazily and the kernel
# evicts cold ones. Wired (non-evictable) memory is roughly:
#
#   ~2 GB Metal VRAM (attention + active layers)
#   ~1 GB anonymous (kv cache + buffers)
#   = ~3 GB true allocation budget
#
# The mmap region is NOT a budget item; it's evictable cache.
# Block only when we genuinely can't fit the ~3 GB allocation:
# kernel says CRITICAL, OR available RAM is below that allocation.
# Anything less is a warning the user can launch through.
_PRESSURE_LEVEL_BLOCK = 3               # ≥ 3 (CRITICAL) = warning; 2 (WARN) is normal load
_MIN_ALLOCATABLE_MB = 500               # advisory only; do not block TUI startup
_REASONABLE_MODEL_HEADROOM_GB = 1.5     # leave this much for the rest of the system
_STUCK_PROCESS_MIN_AGE_SEC = 60         # D-state proc older than this = really stuck


@dataclass(frozen=True)
class SystemHealth:
    ok: bool
    # Human-readable explanation. Shown to the user verbatim when `.ok`
    # is False. Empty when `.ok` is True.
    message: str
    # Structured details for logging / telemetry.
    available_ram_mb: int
    pressure_level: int
    swap_used_fraction: float
    compressor_gb: float
    stuck_servers: list[dict]


def check_system_health() -> SystemHealth:
    """Top-level preflight. Combines memory + stuck-server checks.

    Ordering: we check stuck-server first because a D-state llama-server
    is a more specific / actionable diagnosis than "your system is
    generically low on memory".
    """
    stuck = find_stuck_servers()
    mem = _memory_state()

    if stuck:
        # The TUI's on_mount will auto-trigger recovery and only show
        # this message if recovery also fails. So the message is the
        # honest end-state: user hit the rare non-recoverable case.
        names = ", ".join(f"pid {s['pid']}" for s in stuck[:3])
        return SystemHealth(
            ok=False,
            message=(
                f"A previous model server ({names}) is stuck in a kernel "
                "wait that automatic recovery couldn't clear. Please restart "
                "your Mac and open LocalCode again."
            ),
            available_ram_mb=mem["available_mb"],
            pressure_level=mem["pressure_level"],
            swap_used_fraction=mem["swap_used_fraction"],
            compressor_gb=mem["compressor_gb"],
            stuck_servers=stuck,
        )

    # Memory signals are advisory. Blocking here made `localcode` fail before
    # the TUI on transient macOS page-accounting dips ("only 500 MB allocatable")
    # even though retrying seconds later worked. Runtime/setup own real model
    # launch failures and can show actionable errors; preflight should only hard
    # stop on unrecoverable stuck server processes.
    #
    # We still return structured memory details so telemetry/diagnostics can
    # explain slow starts and OOMs.
    #
    # Do NOT block on:
    #   - kernel CRITICAL pressure;
    #   - low allocatable RAM;
    #   - swap/compressor fullness.
    #
    # The mmap'd model itself doesn't need to "fit" in available RAM
    # because pages fault in lazily and the kernel evicts cold ones.

    return SystemHealth(
        ok=True,
        message="",
        available_ram_mb=mem["available_mb"],
        pressure_level=mem["pressure_level"],
        swap_used_fraction=mem["swap_used_fraction"],
        compressor_gb=mem["compressor_gb"],
        stuck_servers=[],
    )


def find_stuck_servers() -> list[dict]:
    """Return info on any llama-server process that's in D-state (uninterruptible
    sleep) and has been there for > _STUCK_PROCESS_MIN_AGE_SEC seconds.

    macOS `ps` state flags: `U` = uninterruptible sleep, `E` = exiting.
    Anything `U*` means the kernel owns the process and no signal can
    reach it. Combined with long age, that's a stuck process we should
    warn the user about.
    """
    try:
        r = subprocess.run(
            ["ps", "-Ao", "pid,state,etime,command"],
            capture_output=True, text=True, timeout=3,
            env=clean_env(),
        )
    except Exception:
        return []

    stuck: list[dict] = []
    for line in r.stdout.splitlines()[1:]:  # skip header
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid_str, state, etime, cmd = parts
        if "llama-server" not in cmd:
            continue
        # U state = uninterruptible sleep. `Us`, `UE`, etc. all qualify.
        if not state.startswith("U"):
            continue
        age_sec = _parse_etime(etime)
        if age_sec is None or age_sec < _STUCK_PROCESS_MIN_AGE_SEC:
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        stuck.append({"pid": pid, "state": state, "age_sec": age_sec, "cmd": cmd[:80]})
    return stuck


def _parse_etime(etime: str) -> int | None:
    """Parse ps's etime format into seconds. Formats:
      MM:SS, HH:MM:SS, DD-HH:MM:SS.
    """
    try:
        days = 0
        if "-" in etime:
            d, etime = etime.split("-", 1)
            days = int(d)
        parts = etime.split(":")
        if len(parts) == 2:  # MM:SS
            mm, ss = parts
            return days * 86400 + int(mm) * 60 + int(ss)
        if len(parts) == 3:  # HH:MM:SS
            hh, mm, ss = parts
            return days * 86400 + int(hh) * 3600 + int(mm) * 60 + int(ss)
    except Exception:
        pass
    return None


def _memory_state() -> dict:
    """Return the bits of vm_stat + swapusage + pressure we care about.

    `available_mb` is what matters on macOS: free + inactive + purgeable
    + speculative. Those page classes are the kernel's "I can give this
    back immediately if something asks" pool. `Pages free` alone is
    misleadingly tiny on a healthy Mac.
    """
    out: dict = {
        "available_mb": 0,
        "pressure_level": 1,
        "swap_used_fraction": 0.0,
        "compressor_gb": 0.0,
    }
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2, env=clean_env()).stdout
        page = 16384  # Apple Silicon
        available_pages = 0
        for line in vm.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            v = v.strip().rstrip(".")
            try:
                pages = int(v)
            except ValueError:
                continue
            k = k.strip()
            if k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                available_pages += pages
            elif k == "Pages occupied by compressor":
                out["compressor_gb"] = (pages * page) / (1024 ** 3)
        out["available_mb"] = (available_pages * page) // (1024 * 1024)
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=2,
            env=clean_env(),
        )
        out["pressure_level"] = int(r.stdout.strip())
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True, text=True, timeout=2,
            env=clean_env(),
        )
        # "total = 11264.00M  used = 10115.44M  free = 1148.56M  (encrypted)"
        tokens = r.stdout.replace(",", " ").split()
        total = used = None
        for i, t in enumerate(tokens):
            if t == "total" and i + 2 < len(tokens):
                total = float(tokens[i + 2].rstrip("M"))
            elif t == "used" and i + 2 < len(tokens):
                used = float(tokens[i + 2].rstrip("M"))
        if total and used is not None and total > 0:
            out["swap_used_fraction"] = used / total
    except Exception:
        pass

    return out


def estimate_fit(model_size_gb: float) -> tuple[bool, str]:
    """Will `model_size_gb` comfortably fit on this Mac? Used by the
    model picker to annotate choices.

    Policy: need (RAM − OS overhead) ≥ model_size + 1.5 GB headroom.
    """
    try:
        import platform
        if platform.system().lower() != "darwin":
            return True, ""
        total_gb = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
            env=clean_env(),
        ).stdout.strip()) / (1024 ** 3)
    except Exception:
        return True, ""
    # Rough OS + apps overhead on macOS.
    os_overhead_gb = 3.0
    usable = total_gb - os_overhead_gb
    if model_size_gb + _REASONABLE_MODEL_HEADROOM_GB <= usable:
        return True, ""
    shortfall = (model_size_gb + _REASONABLE_MODEL_HEADROOM_GB) - usable
    return False, (
        f"this model needs ~{model_size_gb + _REASONABLE_MODEL_HEADROOM_GB:.1f} GB "
        f"but your Mac has ~{usable:.1f} GB for apps after OS "
        f"(short by ~{shortfall:.1f} GB) — expect swap thrashing"
    )
