"""Dedicated context-window ceiling for every Apple Silicon RAM variant.

Validated anchors (turbo: 16→64K, 48→96K, 64→128K; cohere: 16→16K, 32→32K,
48→48K, 128→64K) are exact; every other tier is interpolated strictly between
its validated neighbours, so no tier exceeds a hardware-validated value.
"""
import pytest

from localcode.model_config import ram_ctx_ceiling, cohere_ctx_ceiling

# Real Apple Silicon RAM configs (MBA/MBP/Studio/Mini).
_VARIANTS = [8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]


@pytest.mark.parametrize("ram,expected", [
    (8, 65536), (16, 65536), (18, 65536), (24, 73728), (32, 81920),
    (36, 90112), (48, 98304), (64, 131072), (96, 131072), (128, 131072),
    (192, 131072),
])
def test_turbo_ceiling_per_variant(ram, expected):
    assert ram_ctx_ceiling(ram) == expected


@pytest.mark.parametrize("ram,expected", [
    (16, 16384), (24, 24576), (32, 32768), (36, 40960), (48, 49152),
    (64, 57344), (96, 65536), (128, 65536),
])
def test_cohere_ceiling_per_variant(ram, expected):
    assert cohere_ctx_ceiling(ram) == expected


def test_validated_anchors_unchanged():
    # These are the on-hardware-validated values; they must never drift.
    assert ram_ctx_ceiling(16) == 65536
    assert ram_ctx_ceiling(48) == 98304
    assert ram_ctx_ceiling(64) == 131072
    assert cohere_ctx_ceiling(48) == 49152


def test_monotonic_non_decreasing():
    for fn in (ram_ctx_ceiling, cohere_ctx_ceiling):
        vals = [fn(r) for r in _VARIANTS]
        assert vals == sorted(vals), f"{fn.__name__} not monotonic: {vals}"


def test_32gb_no_longer_lumped_with_16gb():
    # The gap this fixes: 32 GB used to get the same 64K as 16 GB.
    assert ram_ctx_ceiling(32) > ram_ctx_ceiling(16)
