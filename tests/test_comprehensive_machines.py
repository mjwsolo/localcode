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
    choice = catalog.recommend(ram)
    assert choice in catalog.CHOICES
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


def test_16gb_mac_gets_the_new_12b():
    """The user's headline case: a 16 GB Mac should land on a model that
    fits its ~55% budget — the Gemma 4 12B Q4 we just added."""
    choice = catalog.recommend(16)
    assert choice.size_gb <= 16 * 0.55
    assert "12b" in choice.key


def test_large_ram_gets_a_large_model():
    """A 128 GB workstation should be offered (close to) the biggest model."""
    biggest = max(catalog.CHOICES, key=lambda c: c.size_gb)
    assert catalog.recommend(128).size_gb >= biggest.size_gb * 0.6


@pytest.mark.parametrize("ram", RAM_LADDER)
def test_recommended_model_fits_budget_or_is_smallest(ram):
    choice = catalog.recommend(ram)
    smallest = min(catalog.CHOICES, key=lambda c: c.size_gb)
    assert choice.size_gb <= ram * 0.55 or choice is smallest


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
        SafetyConfig, SearchConfig, UIConfig,
    )
    # No explicit model/profile → eligible for auto-promotion.
    return AppConfig(
        runtime=RuntimeConfig(profile="", model=""),
        search=SearchConfig(), ui=UIConfig(),
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
