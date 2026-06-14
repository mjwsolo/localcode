"""Thermal awareness for sustained local inference (detection + advisory).

Sustained local inference pegs the GPU/CPU and the Mac runs hot; once
macOS starts throttling, tokens/sec quietly drops. We read a thermal
signal so callers can OPTIONALLY back off BEFORE the OS forces it.

Signal source: `pmset -g therm` — the only no-sudo, no-dependency way to
read thermal state on Apple Silicon. (`powermetrics` needs sudo; the
IOKit thermal-pressure notification needs a native dep.) Its key line is
`CPU_Speed_Limit = N` where N is a percentage: 100 = no throttle, <100 =
the OS is capping clocks for heat. When the machine is cool, pmset prints
"No thermal warning level has been recorded" and omits CPU_Speed_Limit
entirely — which we treat as nominal.

We map the speed-limit percentage onto the same vocabulary Apple's own
NSProcessInfoThermalState uses, so the levels are familiar:
  nominal  — 100%      no throttling
  fair     — 75–99%    mild throttling beginning
  serious  — 50–74%    notable throttling
  critical — <50%      heavy throttling, OS is actively protecting HW

NOTHING here changes runtime behaviour. These are advisory readings;
wiring them into the server launch is a runtime.py change that is only
DESCRIBED in the ticket report, never performed from this module.

This lives in its own module (rather than performance.py) purely to keep
performance.py within its file-size budget; performance.py re-exports
these names for a stable import path.
"""
from __future__ import annotations

from dataclasses import dataclass
import platform
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # MachineProfile is only referenced in annotations.
    from .performance import MachineProfile

_THERMAL_LEVELS = ("nominal", "fair", "serious", "critical")


def _thermal_level_from_speed_limit(percent: int) -> str:
    """Map a CPU_Speed_Limit percentage onto a thermal-state level."""
    if percent >= 100:
        return "nominal"
    if percent >= 75:
        return "fair"
    if percent >= 50:
        return "serious"
    return "critical"


def read_cpu_speed_limit() -> int:
    """Raw CPU_Speed_Limit percentage from `pmset -g therm` (0–100).

    100 means the OS is imposing no thermal speed cap. Returns 100
    (the safe, "not throttled" default) on non-Darwin platforms, when
    pmset is missing, on timeout/parse failure, or when pmset reports
    no thermal warning (the cool-machine case where the field is
    absent). Dependency-free; short timeout; never raises.
    """
    if platform.system().lower() != "darwin":
        return 100
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except Exception:
        return 100
    text = result.stdout or ""
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    if not match:
        # No CPU_Speed_Limit line → machine is cool (pmset prints the
        # "No thermal warning level has been recorded" notes instead).
        return 100
    try:
        value = int(match.group(1))
    except ValueError:
        return 100
    # Clamp defensively — the field is a percentage.
    return max(0, min(100, value))


def read_thermal_pressure() -> str:
    """Current thermal pressure as a level string.

    One of: "nominal", "fair", "serious", "critical" (increasing heat).
    Dependency-free, reads `pmset -g therm` on Apple Silicon, and
    returns "nominal" (the safe default) on any non-Darwin platform or
    on any read/parse failure. Never raises.
    """
    return _thermal_level_from_speed_limit(read_cpu_speed_limit())


@dataclass(slots=True)
class ThermalCaps:
    """Advisory, conservative caps suggested for the current heat level.

    ADVISORY ONLY — these are recommendations a caller may apply when
    launching/sizing work. Nothing in this module wires them into the
    runtime. Fields:
      pressure         the level these caps respond to
      throttle         True once any backing-off is advised (fair+)
      thread_scale     multiplier (≤1.0) to apply to llama_cpp_threads
      batch_scale      multiplier (≤1.0) to apply to llama_cpp_batch_size
      cooldown_seconds suggested pause between requests (0 = none)
      reason           human-readable explanation
    """

    pressure: str
    throttle: bool
    thread_scale: float
    batch_scale: float
    cooldown_seconds: int
    reason: str


def recommended_thermal_caps(
    pressure: str | None = None,
    machine: MachineProfile | None = None,
) -> ThermalCaps:
    """Advisory caps to ease thermal load at the given pressure level.

    Conservative by design: at "nominal" it suggests no change at all.
    As pressure rises it suggests progressively smaller thread/batch
    multipliers and a short inter-request cooldown so the machine can
    shed heat. These are SUGGESTIONS — callers decide whether to honour
    them; see the runtime hook described in the ticket report. Pure
    function (no side effects); reads live thermal state only when
    `pressure` is omitted.

    `machine` is accepted for future tier-aware tuning and to mirror the
    other advisory helpers; small/medium (laptop) machines get a touch
    more headroom shaved off since they have the least cooling.
    """
    if pressure is None:
        pressure = read_thermal_pressure()
    pressure = str(pressure).strip().lower()
    if pressure not in _THERMAL_LEVELS:
        pressure = "nominal"

    laptop = bool(machine and machine.tier in {"small", "medium"})

    # Cap values live in the central per-Mac config (model_config.THERMAL_CAPS);
    # this assembles the ThermalCaps dataclass from that table so the levels
    # are edited in one place.
    from .model_config import THERMAL_CAPS
    spec = THERMAL_CAPS[pressure]
    thread_scale = spec["thread_scale_laptop"] if laptop else spec["thread_scale_other"]
    return ThermalCaps(
        pressure=pressure,
        throttle=bool(spec["throttle"]),
        thread_scale=float(thread_scale),
        batch_scale=float(spec["batch_scale"]),
        cooldown_seconds=int(spec["cooldown_seconds"]),
        reason=str(spec["reason"]),
    )
