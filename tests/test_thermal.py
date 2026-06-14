"""Thermal awareness: detection (pmset parse) + advisory caps.

We can't make CI run hot, but the whole feature is a pure function of
the `pmset -g therm` output plus a pressure level, so we drive it with
mocked pmset text across the cool / mild / heavy / parse-failure cases
and assert the parse + the advisory both behave conservatively.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import performance as perf
from localcode import thermal


def test_performance_reexports_thermal_api():
    """The thermal helpers stay importable from performance.py (their
    historical home) even though they now live in localcode.thermal."""
    assert perf.read_thermal_pressure is thermal.read_thermal_pressure
    assert perf.recommended_thermal_caps is thermal.recommended_thermal_caps
    assert perf.read_cpu_speed_limit is thermal.read_cpu_speed_limit
    assert perf.ThermalCaps is thermal.ThermalCaps


def _fake_pmset(stdout: str):
    """Return a callable that mimics subprocess.run for `pmset -g therm`."""
    return lambda *a, **kw: type("R", (), {"stdout": stdout, "stderr": "", "returncode": 0})()


# Real cool-machine output: pmset omits CPU_Speed_Limit entirely.
_COOL_OUTPUT = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "Note: No CPU power status has been recorded\n"
)

# Throttling output: CPU_Speed_Limit present and below 100.
_THROTTLED_OUTPUT = (
    "Note: Thermal pressure may cause performance reduction.\n"
    "CPU_Power_Notify\t\t = 0x1\n"
    "CPU_Speed_Limit \t\t = 60\n"
)


# ── read_cpu_speed_limit / read_thermal_pressure ───────────────────


def test_speed_limit_parsed_from_pmset(monkeypatch):
    monkeypatch.setattr(thermal.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(thermal.subprocess, "run", _fake_pmset(_THROTTLED_OUTPUT))
    assert thermal.read_cpu_speed_limit() == 60
    assert thermal.read_thermal_pressure() == "serious"


def test_cool_machine_is_nominal(monkeypatch):
    """No CPU_Speed_Limit line → treat as fully nominal (not throttled)."""
    monkeypatch.setattr(thermal.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(thermal.subprocess, "run", _fake_pmset(_COOL_OUTPUT))
    assert thermal.read_cpu_speed_limit() == 100
    assert thermal.read_thermal_pressure() == "nominal"


@pytest.mark.parametrize("percent,level", [
    (100, "nominal"),
    (99, "fair"),
    (75, "fair"),
    (74, "serious"),
    (50, "serious"),
    (49, "critical"),
    (0, "critical"),
])
def test_level_mapping(percent, level, monkeypatch):
    monkeypatch.setattr(thermal.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        thermal.subprocess, "run",
        _fake_pmset(f"CPU_Speed_Limit = {percent}\n"),
    )
    assert thermal.read_cpu_speed_limit() == percent
    assert thermal.read_thermal_pressure() == level


def test_value_clamped(monkeypatch):
    monkeypatch.setattr(thermal.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(thermal.subprocess, "run", _fake_pmset("CPU_Speed_Limit = 250\n"))
    assert thermal.read_cpu_speed_limit() == 100


def test_non_darwin_is_safe_default(monkeypatch):
    monkeypatch.setattr(thermal.platform, "system", lambda: "Linux")
    # subprocess must not even be called on non-Darwin.
    def _boom(*a, **kw):
        raise AssertionError("pmset should not run off Darwin")
    monkeypatch.setattr(thermal.subprocess, "run", _boom)
    assert thermal.read_cpu_speed_limit() == 100
    assert thermal.read_thermal_pressure() == "nominal"


def test_pmset_missing_falls_back(monkeypatch):
    monkeypatch.setattr(thermal.platform, "system", lambda: "Darwin")
    def _missing(*a, **kw):
        raise FileNotFoundError("pmset")
    monkeypatch.setattr(thermal.subprocess, "run", _missing)
    assert thermal.read_cpu_speed_limit() == 100
    assert thermal.read_thermal_pressure() == "nominal"


def test_pmset_timeout_falls_back(monkeypatch):
    monkeypatch.setattr(thermal.platform, "system", lambda: "Darwin")
    import subprocess as _sp
    def _timeout(*a, **kw):
        raise _sp.TimeoutExpired(cmd="pmset", timeout=3)
    monkeypatch.setattr(thermal.subprocess, "run", _timeout)
    assert thermal.read_thermal_pressure() == "nominal"


def test_garbage_output_falls_back(monkeypatch):
    monkeypatch.setattr(thermal.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(thermal.subprocess, "run", _fake_pmset("totally unrelated text\n"))
    assert thermal.read_cpu_speed_limit() == 100
    assert thermal.read_thermal_pressure() == "nominal"


# ── recommended_thermal_caps ───────────────────────────────────────


def test_nominal_caps_are_no_op():
    caps = thermal.recommended_thermal_caps("nominal")
    assert caps.throttle is False
    assert caps.thread_scale == 1.0
    assert caps.batch_scale == 1.0
    assert caps.cooldown_seconds == 0


def test_caps_get_more_conservative_with_pressure():
    """Higher pressure must never suggest MORE load than lower pressure."""
    levels = ["nominal", "fair", "serious", "critical"]
    caps = [thermal.recommended_thermal_caps(p) for p in levels]

    threads = [c.thread_scale for c in caps]
    batches = [c.batch_scale for c in caps]
    cooldowns = [c.cooldown_seconds for c in caps]

    assert threads == sorted(threads, reverse=True), threads
    assert batches == sorted(batches, reverse=True), batches
    assert cooldowns == sorted(cooldowns), cooldowns
    # Everything above nominal advises throttling.
    assert [c.throttle for c in caps] == [False, True, True, True]


def test_caps_scales_stay_in_bounds():
    for p in ["nominal", "fair", "serious", "critical"]:
        caps = thermal.recommended_thermal_caps(p)
        assert 0.0 < caps.thread_scale <= 1.0
        assert 0.0 < caps.batch_scale <= 1.0
        assert caps.cooldown_seconds >= 0


def test_unknown_pressure_treated_as_nominal():
    caps = thermal.recommended_thermal_caps("banana")
    assert caps.pressure == "nominal"
    assert caps.throttle is False


def test_laptop_gets_extra_headroom_shaved():
    """Small/medium (laptop) machines have the least cooling, so under the
    same pressure they should be throttled at least as hard as a big box."""
    laptop = perf.MachineProfile(
        system="darwin", cpu_cores=8, memory_gb=16,
        gpu_summary="Apple M3", has_gpu=True, tier="small",
    )
    workstation = perf.MachineProfile(
        system="darwin", cpu_cores=24, memory_gb=128,
        gpu_summary="Apple M3 Ultra", has_gpu=True, tier="workstation",
    )
    lap = thermal.recommended_thermal_caps("critical", laptop)
    work = thermal.recommended_thermal_caps("critical", workstation)
    assert lap.thread_scale <= work.thread_scale


def test_caps_reads_live_pressure_when_omitted(monkeypatch):
    monkeypatch.setattr(thermal, "read_thermal_pressure", lambda: "serious")
    caps = thermal.recommended_thermal_caps()
    assert caps.pressure == "serious"
    assert caps.throttle is True
