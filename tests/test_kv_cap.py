"""RAM-aware KV-cache cap — the model-agnostic OOM never-again guard.

The cohere/North-Mini OOM (`zsh: killed`) was a too-large `--ctx-size`
allocating a multi-tens-of-GB uncompressed-f16 KV cache. The per-model
`_cohere_ctx_ceiling` was a point-fix; `_kv_aware_ctx_ceiling` generalizes it:
never launch a context whose KV cache won't fit in RAM, for ANY model.

These tests assert the invariant (KV + weights + reserve fits) across every
RAM tier, AND that the cap defers (no-op) when KV size isn't confidently
computable — so the validated turbo-compressed ceilings are never regressed.
"""
from __future__ import annotations

import pytest

from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway

RAM_TIERS = [8, 16, 24, 32, 36, 48, 64, 96, 128, 192]
# A heavy attention shape (≈ a 30B MoE): drives a large per-token KV.
HEAVY_SHAPE = {"n_layers": 48, "n_kv_heads": 8, "head_dim": 128}


def _gw():
    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.model = "north-mini-code-30b-a3b.gguf"  # name triggers cohere detection
    return LocalCodeRuntimeGateway(cfg)


def _reserve(total: int) -> int:
    return max(3 * 1024 ** 3, int(total * 0.15))


def test_kv_bytes_per_token_f16_matches_formula(monkeypatch):
    gw = _gw()
    monkeypatch.setattr(gw, "_gguf_kv_meta", lambda mp=None: dict(HEAVY_SHAPE))
    monkeypatch.setattr(gw, "_effective_kv_dtypes", lambda mp: ("f16", "f16"))
    bpt = gw._kv_bytes_per_token()
    assert bpt == 48 * 8 * 128 * (2.0 + 2.0)


def test_cap_defers_on_unknown_compressed_dtype(monkeypatch):
    # Turbo-compressed KV (unknown exact bytes) -> bytes/token None -> the cap
    # is a no-op (large sentinel) so the validated ceilings govern. This is the
    # no-regression guarantee for the TurboQuant path.
    gw = _gw()
    monkeypatch.setattr(gw, "_gguf_kv_meta", lambda mp=None: dict(HEAVY_SHAPE))
    monkeypatch.setattr(gw, "_effective_kv_dtypes", lambda mp: ("q8_0", "turbo4"))
    assert gw._kv_bytes_per_token() is None
    assert gw._kv_aware_ctx_ceiling(None, 128) >= 1_000_000


def test_kv_cap_invariant_holds_across_all_ram_tiers(monkeypatch):
    # The core invariant: for a heavy f16 model, the capped context's KV cache
    # PLUS weights PLUS reserve must fit in RAM, at EVERY tier.
    gw = _gw()
    weights = 20 * 1024 ** 3  # ~20 GB (Q8 30B)
    bpt = 48 * 8 * 128 * 4.0  # f16 K+V
    monkeypatch.setattr(gw, "_kv_bytes_per_token", lambda mp=None: bpt)
    monkeypatch.setattr(gw, "_model_file_bytes", lambda mp=None: weights)
    for ram in RAM_TIERS:
        total = ram * 1024 ** 3
        cap = gw._kv_aware_ctx_ceiling(None, ram)
        assert cap >= 2048
        assert cap % 2048 == 0, f"ctx must be a 2048 multiple (ram={ram})"
        if total - weights - _reserve(total) > 0:
            kv_at_cap = cap * bpt
            assert kv_at_cap + weights + _reserve(total) <= total, (
                f"KV+weights+reserve overflows RAM at {ram}GB: "
                f"cap={cap}, kv={kv_at_cap/1e9:.1f}GB"
            )


def test_cap_monotonic_in_ram(monkeypatch):
    gw = _gw()
    monkeypatch.setattr(gw, "_kv_bytes_per_token", lambda mp=None: 48 * 8 * 128 * 4.0)
    monkeypatch.setattr(gw, "_model_file_bytes", lambda mp=None: 12 * 1024 ** 3)
    caps = [gw._kv_aware_ctx_ceiling(None, r) for r in RAM_TIERS]
    assert caps == sorted(caps), f"cap must be monotonic in RAM: {caps}"


def test_target_num_ctx_applies_cap(monkeypatch):
    # A path that would return 256K gets clamped down to the KV-fit ceiling.
    gw = _gw()
    monkeypatch.setattr(gw, "_system_ram_gb", lambda: 16)
    monkeypatch.setattr(gw, "_target_num_ctx_uncapped", lambda o=None, mp=None: 262144)
    monkeypatch.setattr(gw, "_kv_aware_ctx_ceiling", lambda mp, ram: 32768)
    assert gw._target_num_ctx() == 32768


def test_target_num_ctx_turbo_path_not_clamped(monkeypatch):
    # When KV size isn't computable (turbo), the cap defers and the validated
    # ceiling (here 128K) is returned unchanged — no regression.
    gw = _gw()
    monkeypatch.setattr(gw, "_system_ram_gb", lambda: 64)
    monkeypatch.setattr(gw, "_target_num_ctx_uncapped", lambda o=None, mp=None: 131072)
    monkeypatch.setattr(gw, "_kv_aware_ctx_ceiling", lambda mp, ram: 1_000_000)
    assert gw._target_num_ctx() == 131072


def test_explicit_override_bypasses_cap(monkeypatch):
    gw = _gw()
    monkeypatch.setattr(gw, "_kv_aware_ctx_ceiling", lambda mp, ram: 4096)
    # An explicit override is a deliberate value and must not be KV-clamped.
    assert gw._target_num_ctx(num_ctx_override=99999) == 99999
