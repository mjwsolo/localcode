"""Multi-machine / RAM-size coverage.

We can't spin up a 128 GB Mac in CI, but every RAM-dependent DECISION is a
pure function of a machine profile or a RAM number — so we drive those
functions across the full Apple-Silicon RAM ladder (8 → 128 GB) and assert
the model recommendation, GPU-offload decision, and profile promotion all
behave sanely. This is the deterministic stand-in for "test it on every
Mac size."
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import models_catalog as catalog
from localcode import performance as perf

# The Apple-Silicon configurations LocalCode actually ships against.
RAM_LADDER = [8, 16, 18, 24, 32, 36, 48, 64, 96, 128]


# ── Model recommendation scales with RAM ────────────────────────────


@pytest.mark.parametrize("ram", RAM_LADDER)
def test_recommend_returns_a_real_catalog_entry(ram):
    """A curated entry, or a quant the vendor repo really ships (recorded size)."""
    choice = catalog.recommend(ram)
    group = catalog.group_for_filename(choice.filename)
    assert choice in catalog.CHOICES or (
        group is not None and choice.filename in catalog.KNOWN_QUANT_BYTES[group.hf_repo])
    assert choice.size_gb > 0


def test_recommendation_is_monotonic_in_ram():
    """More RAM must never recommend a *smaller* model."""
    sizes = [catalog.recommend(r).size_gb for r in RAM_LADDER]
    assert sizes == sorted(sizes), f"non-monotonic recommendation: {sizes}"


def test_small_ram_gets_smallest_model():
    """An 8 GB machine can't fit any catalog model in budget → it must be
    offered the smallest one, never something it can't load."""
    smallest = min(catalog.CHOICES, key=lambda c: c.size_gb)
    assert catalog.recommend(8) is smallest


def test_16gb_mac_gets_the_12b_with_its_drafter():
    """A 16 GB Mac (10.7 GiB GPU working set) lands on Gemma 4 12B UD-Q5_K_XL:
    no 26B/35B quant above the degraded line fits beside a working context,
    and Q5 + the 0.47 GB MTP drafter still does."""
    choice = catalog.recommend(16)
    assert choice.filename == "gemma-4-12b-it-UD-Q5_K_XL.gguf"
    plan = catalog.quant_plan(catalog.group_for_filename(choice.filename), choice.size_gb, 16)
    assert plan["fit"] == "fits" and plan["drafter"]


def test_large_ram_gets_a_large_model():
    """A 128 GB workstation should be offered (close to) the biggest model."""
    biggest = max(catalog.CHOICES, key=lambda c: c.size_gb)
    assert catalog.recommend(128).size_gb >= biggest.size_gb * 0.6


@pytest.mark.parametrize("ram", RAM_LADDER)
def test_recommended_model_fits_and_is_not_degraded(ram):
    choice = catalog.recommend(ram)
    group = catalog.group_for_filename(choice.filename)
    plan = catalog.quant_plan(group, choice.size_gb, ram)
    if plan["fit"] != "fits":  # only when nothing fits: the smallest runnable model
        assert all(catalog.recommended_quant_index(catalog.known_quants(g), g, ram) is None
                   for g in catalog._groups_by_capability())
        return
    assert not catalog.is_degraded(catalog._quant_label(choice.filename), group)


# ── GPU-offload decision per RAM ────────────────────────────────────


def test_metal_offload_off_below_16gb(monkeypatch):
    monkeypatch.setattr(perf.platform, "system", lambda: "Darwin")
    assert perf.metal_gpu_available(8) is False
    assert perf.metal_gpu_available(12) is False


def test_metal_offload_on_at_32gb_plus(monkeypatch):
    monkeypatch.setattr(perf.platform, "system", lambda: "Darwin")
    assert perf.metal_gpu_available(32) is True
    assert perf.metal_gpu_available(128) is True


def test_metal_offload_off_on_non_mac(monkeypatch):
    monkeypatch.setattr(perf.platform, "system", lambda: "Linux")
    assert perf.metal_gpu_available(64) is False


# ── Legacy-default promotion across machine tiers ───────────────────


def _machine(memory_gb, tier, *, system="darwin", gpu=True):
    return perf.MachineProfile(
        system=system, cpu_cores=10, memory_gb=memory_gb,
        gpu_summary="Apple M-series", has_gpu=gpu, tier=tier,
    )


def _blank_config():
    from localcode.config import (
        AppConfig, LoggingConfig, RuntimeConfig,
        SafetyConfig, UIConfig,
    )
    # No explicit model/profile → eligible for auto-promotion.
    return AppConfig(
        runtime=RuntimeConfig(profile="", model=""),
        ui=UIConfig(),
        safety=SafetyConfig(), logging=LoggingConfig(),
    )


@pytest.mark.parametrize("tier,expected", [
    ("small", True), ("medium", True), ("large", False),
])
def test_promotion_depends_on_tier(tier, expected):
    machine = _machine(16 if tier == "small" else 64, tier)
    assert perf.should_promote_legacy_default_to_laptop_26b(_blank_config(), machine) is expected


def test_no_promotion_without_gpu():
    machine = _machine(16, "small", gpu=False)
    assert perf.should_promote_legacy_default_to_laptop_26b(_blank_config(), machine) is False


def test_no_promotion_on_non_mac():
    machine = _machine(64, "small", system="linux")
    assert perf.should_promote_legacy_default_to_laptop_26b(_blank_config(), machine) is False


# ── apple_silicon_bandwidth_gbps chip table ─────────────────────────


@pytest.mark.parametrize("brand,expected", [
    # M1 family
    ("Apple M1",           68.0),
    ("Apple M1 Pro",      200.0),
    ("Apple M1 Max",      400.0),
    ("Apple M1 Ultra",    800.0),
    # M2 family
    ("Apple M2",          100.0),
    ("Apple M2 Pro",      200.0),
    ("Apple M2 Max",      400.0),
    ("Apple M2 Ultra",    800.0),
    # M3 family — M3 Pro is 150, NOT 200 (regression vs M2 Pro)
    ("Apple M3",          100.0),
    ("Apple M3 Pro",      150.0),
    ("Apple M3 Max",      400.0),
    ("Apple M3 Ultra",    819.0),
    # M4 family (no M4 Ultra released)
    ("Apple M4",          120.0),
    ("Apple M4 Pro",      273.0),
    ("Apple M4 Max",      546.0),
    # M5 family
    ("Apple M5",          153.0),
    ("Apple M5 Pro",      307.0),
    ("Apple M5 Max",      614.0),
    ("Apple M5 Ultra",   1100.0),
])
def test_bandwidth_table(brand, expected, monkeypatch):
    """Each chip tier must map to its documented peak bandwidth."""
    monkeypatch.setattr(perf.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        perf.subprocess, "run",
        lambda *a, **kw: type("R", (), {"stdout": brand, "returncode": 0})(),
    )
    result = perf.apple_silicon_bandwidth_gbps()
    assert result == expected, (
        f"{brand}: expected {expected} GB/s, got {result} GB/s"
    )


def test_bandwidth_fallback_non_mac(monkeypatch):
    monkeypatch.setattr(perf.platform, "system", lambda: "Linux")
    assert perf.apple_silicon_bandwidth_gbps() == 150.0
