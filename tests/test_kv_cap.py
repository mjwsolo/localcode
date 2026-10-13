"""RAM-aware KV-cache cap — the model-agnostic OOM never-again guard.

The cohere/North-Mini OOM (`zsh: killed`) was a too-large `--ctx-size`
allocating a multi-tens-of-GB uncompressed-f16 KV cache. The per-model
`_cohere_ctx_ceiling` was a point-fix; `_kv_aware_ctx_ceiling` generalizes it:
never launch a context whose KV cache won't fit in RAM, for ANY model.

These tests assert the invariant (KV + weights + reserve fits) across every
RAM tier, AND that the cap defers (no-op) when KV size isn't confidently
computable. Known TurboQuant formats include padding and adaptive layers.
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


def test_cap_defers_on_unknown_dtype(monkeypatch):
    # Unsupported storage formats defer to the independent RAM ceiling.
    gw = _gw()
    monkeypatch.setattr(gw, "_gguf_kv_meta", lambda mp=None: dict(HEAVY_SHAPE))
    monkeypatch.setattr(gw, "_effective_kv_dtypes", lambda mp: ("q8_0", "unknown"))
    assert gw._kv_bytes_per_token() is None
    assert gw._kv_aware_ctx_ceiling(None, 128) >= 1_000_000


@pytest.mark.parametrize("cache,block_bytes", [("turbo2",34),("turbo3",50),("turbo4",68)])
def test_turbo_cache_uses_real_storage_and_distinct_value_width(monkeypatch, cache, block_bytes):
    monkeypatch.setenv("TURBO_LAYER_ADAPTIVE", "0")
    gw = _gw()
    monkeypatch.setattr(gw, "_gguf_kv_meta", lambda mp=None: {**HEAVY_SHAPE, "value_head_dim": 256})
    monkeypatch.setattr(gw, "_effective_kv_dtypes", lambda mp: ("q8_0", cache))
    assert gw._kv_bytes_per_token() == 48 * 8 * (128 * 34 / 32 + 256 * block_bytes / 128)
    monkeypatch.setattr(gw, "_model_file_bytes", lambda mp=None: 12 * 1024**3)
    cap = gw._kv_aware_ctx_ceiling(None, 16)
    assert cap * gw._kv_bytes_per_token() + 15 * 1024**3 <= 16 * 1024**3


def test_turbo_padding_and_default_boundary_layers_are_budgeted(monkeypatch):
    monkeypatch.delenv("TURBO_LAYER_ADAPTIVE", raising=False)
    gw = _gw()
    monkeypatch.setattr(gw, "_gguf_kv_meta", lambda mp=None: {**HEAVY_SHAPE, "value_head_dim": 192})
    monkeypatch.setattr(gw, "_effective_kv_dtypes", lambda mp: ("q8_0", "turbo2"))
    assert gw._kv_bytes_per_token() == 8 * (48 * 128 * 34 / 32 + 44 * 256 * 34 / 128 + 4 * 256 * 34 / 32)


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


def test_cache_storage_table_matches_compiled_backend_blocks(tmp_path):
    from pathlib import Path
    import shutil
    import subprocess
    from localcode.model_config import KV_DTYPE_BYTES
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    source = tmp_path / 'storage.cpp'
    source.write_text(r"""
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <cstdio>
int main() {
    std::printf("%zu %zu %zu %zu %d %d %d %d", sizeof(block_q8_0),sizeof(block_turbo2_0),sizeof(block_turbo3_0),sizeof(block_turbo4_0),QK8_0,QK_TURBO2,QK_TURBO3,QK_TURBO4);
}
""")
    binary = tmp_path / 'storage'
    root = Path(__file__).resolve().parents[1]
    subprocess.run([compiler,'-std=c++17','-I'+str(root/'llama-cpp-turboquant/ggml/src'),str(source),'-o',str(binary)], check=True,capture_output=True,timeout=60)
    values = [int(x) for x in subprocess.check_output([str(binary)],text=True,timeout=10).split()]
    for index, dtype in enumerate(('q8_0','turbo2','turbo3','turbo4')):
        assert KV_DTYPE_BYTES[dtype] == values[index] / values[index+4]
