"""GPU working set, KV geometry and the fit rule behind the picker and launcher.

The numbers here are the real ones: Unsloth file sizes recorded in
models_catalog.KNOWN_QUANT_BYTES and KV layouts read from the GGUF headers.
"""
from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from localcode import gpu_memory
from localcode import models_catalog as catalog
from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway

GIB = 1024 ** 3


# ── GPU working set ──────────────────────────────────────────────────


def test_default_fraction_two_thirds_small_three_quarters_large():
    assert gpu_memory.default_fraction(16) == pytest.approx(2 / 3)
    assert gpu_memory.default_fraction(36) == pytest.approx(2 / 3)
    assert gpu_memory.default_fraction(48) == pytest.approx(0.75)
    assert gpu_memory.working_set_bytes(16) == int(16 * GIB * 2 / 3)


def test_live_limit_wins_on_this_machine(monkeypatch):
    monkeypatch.delenv("LOCALCODE_RAM_GB", raising=False)
    monkeypatch.setattr(gpu_memory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu_memory, "_physical_ram_bytes", lambda: 16 * GIB)
    monkeypatch.setattr(gpu_memory, "_sysctl_wired_limit_bytes", lambda: None)
    monkeypatch.setattr(gpu_memory, "probe_metal_working_set", lambda memsize: 11_000 * 1024 ** 2)
    assert gpu_memory.working_set_bytes(16) == 11_000 * 1024 ** 2
    # A user-set iogpu.wired_limit_mb is read (never written).
    monkeypatch.setattr(gpu_memory, "_LIVE", {})
    monkeypatch.setattr(gpu_memory, "_sysctl_wired_limit_bytes", lambda: 12_000 * 1024 ** 2)
    assert gpu_memory.working_set_bytes(16) == 12_000 * 1024 ** 2
    # Another tier (an explicit ram_gb that is not this Mac) uses the default.
    assert gpu_memory.working_set_bytes(24) == int(24 * GIB * 2 / 3)


def test_emulated_tier_ignores_the_live_limit(monkeypatch):
    monkeypatch.setenv("LOCALCODE_RAM_GB", "16")
    monkeypatch.setattr(gpu_memory, "_physical_ram_bytes", lambda: 16 * GIB)
    monkeypatch.setattr(gpu_memory, "_sysctl_wired_limit_bytes", lambda: 14 * GIB)
    assert gpu_memory.working_set_bytes() == int(16 * GIB * 2 / 3)


def test_metal_probe_reads_the_device_list_and_caches_it(monkeypatch, tmp_path):
    monkeypatch.undo()  # the autouse fixture stubs the probe; test the real one
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path))
    binary = tmp_path / "llama-server"
    binary.write_text("")
    import localcode.bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_turboquant_binary_path", lambda: binary)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="Available devices:\n  MTL0: Apple M4 (10923 MiB, 10922 MiB free)\n", stderr="")
    monkeypatch.setattr(gpu_memory.subprocess, "run", run)
    assert gpu_memory.probe_metal_working_set(16 * GIB) == 10923 * 1024 ** 2
    assert gpu_memory.probe_metal_working_set(16 * GIB) == 10923 * 1024 ** 2
    assert len(calls) == 1 and calls[0][1] == "--list-devices"


def test_memory_guard_cap_uses_the_working_set(monkeypatch, tmp_path):
    from localcode import memory_guard
    model = tmp_path / "m.gguf"
    with model.open("wb") as f:
        f.truncate(int(6.2 * GIB))
    monkeypatch.setattr(memory_guard.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout=str(16 * GIB), stderr=""))
    monkeypatch.setenv("LOCALCODE_RAM_GB", "16")
    # 6.2 GiB is under 55% of a flat 75% cap (6.6 GiB) but over 55% of the real 2/3 cap (5.9 GiB).
    assert memory_guard.should_disable_metal_residency(str(model), 8192)


# ── KV geometry from GGUF headers ────────────────────────────────────


def _gguf(path: Path, arch: str, kv: dict) -> Path:
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    def value(v):
        if isinstance(v, bool):
            return struct.pack("<I", 7) + struct.pack("<?", v)
        if isinstance(v, int):
            return struct.pack("<I", 4) + struct.pack("<I", v)
        if isinstance(v, str):
            return struct.pack("<I", 8) + s(v)
        if isinstance(v, list):
            et, fmt = (7, "<?") if isinstance(v[0], bool) else (4, "<I")
            return struct.pack("<I", 9) + struct.pack("<I", et) + struct.pack("<Q", len(v)) + b"".join(struct.pack(fmt, x) for x in v)
        raise TypeError(v)
    items = {"general.architecture": arch, **{f"{arch}.{k}": v for k, v in kv.items()}}
    body = b"".join(s(k) + value(v) for k, v in items.items())
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(items)) + body)
    return path


def _gw(kt="q8_0", vt="q8_0"):
    cfg = RuntimeConfig()
    cfg.kv_cache_type_k, cfg.kv_cache_type_v = kt, vt
    return LocalCodeRuntimeGateway(cfg)


def test_hybrid_model_counts_only_attention_layers(tmp_path):
    """Qwen 3.6: 40 blocks, every 4th is attention. server.log measured
    "K (q8_0): 680.00 MiB" at 131072 cells over 10 layers."""
    p = _gguf(tmp_path / "qwen.gguf", "qwen35moe", {
        "block_count": 40, "attention.head_count": 16, "attention.head_count_kv": 2,
        "attention.key_length": 256, "attention.value_length": 256, "full_attention_interval": 4})
    gw = _gw()
    assert gw._kv_bytes_per_token(str(p)) == 10880
    assert gw._kv_bytes_per_token(str(p)) * 131072 / 2 == 680 * 1024 ** 2  # K half
    assert gw._kv_fixed_bytes(str(p)) == 0


def test_sliding_window_model_counts_global_layers_and_a_fixed_window(tmp_path):
    """Gemma 4 26B-A4B: 5 global layers (2 KV heads x 512) grow with the
    context; 25 sliding-window layers (8 heads x 256) hold n_swa + ubatch cells."""
    pattern = [i % 6 != 5 for i in range(30)]
    p = _gguf(tmp_path / "gemma.gguf", "gemma4", {
        "block_count": 30, "attention.head_count": 16,
        "attention.head_count_kv": [2 if not swa else 8 for swa in pattern],
        "attention.key_length": 512, "attention.value_length": 512,
        "attention.key_length_swa": 256, "attention.value_length_swa": 256,
        "attention.sliding_window": 1024, "attention.sliding_window_pattern": pattern})
    gw = _gw()
    assert gw._kv_bytes_per_token(str(p)) == 5 * 2 * 1024 * 34 / 32
    assert gw._kv_fixed_bytes(str(p)) == 1536 * 25 * 8 * 512 * 34 / 32
    group = catalog.by_group("gemma-4-26b-a4b")
    assert group.kv_bytes_per_token == gw._kv_bytes_per_token(str(p))
    assert group.kv_fixed_bytes == gw._kv_fixed_bytes(str(p))


def test_plain_model_keeps_the_full_layer_formula(tmp_path):
    p = _gguf(tmp_path / "dense.gguf", "llama", {
        "block_count": 32, "attention.head_count": 32, "attention.head_count_kv": 8,
        "attention.key_length": 128, "attention.value_length": 128})
    assert _gw("f16", "f16")._kv_bytes_per_token(str(p)) == 32 * 8 * 256 * 2


# ── Fit, degraded quants and the per-tier recommendation ─────────────


def test_fit_badges():
    bpt = 10880
    assert gpu_memory.fit(int(13.68e9), 24, bpt) == "fits"
    assert gpu_memory.fit(int(16.85e9), 24, bpt) == "tight"
    assert gpu_memory.fit(int(22.36e9), 24, bpt) == "too_big"


@pytest.mark.parametrize("label,degraded", [
    ("UD-IQ1_M", True), ("UD-IQ2_XXS", True), ("UD-IQ2_S", True), ("UD-IQ2_M", True),
    ("IQ2-mix", True), ("UD-Q2_K_XL", False), ("UD-IQ3_XXS", False), ("Q8_0", False), ("BF16", False)])
def test_degraded_below_q2_k_xl_for_models_under_40b(label, degraded):
    assert catalog.is_degraded(label, catalog.by_group("qwen-3.6-35b-a3b")) is degraded


def test_iq2_of_a_large_model_is_not_degraded():
    assert not catalog.is_degraded("UD-IQ2_M", catalog.by_group("kolibri-1"))


def _star(group_key: str, ram: int):
    g = catalog.by_group(group_key)
    rows = catalog.known_quants(g)
    i = catalog.recommended_quant_index(rows, g, ram)
    return None if i is None else catalog._quant_label(rows[i][0])


@pytest.mark.parametrize("group,ram,label", [
    ("gemma-4-12b", 16, "UD-Q5_K_XL"),       # 8.61 GB + 0.47 GB drafter
    ("qwen-3.6-35b-a3b", 16, None),          # IQ1_M (10.05 GB) does not fit 10.7 GiB with a context
    ("qwen-3.8-27b", 16, None),              # only the degraded IQ2_S fits: no star
    ("qwen-3.6-35b-a3b", 24, "UD-IQ3_S"),    # was UD-Q2_K_XL under 0.55 x RAM
    ("gemma-4-26b-a4b", 24, "UD-IQ4_NL"),
    ("qwen-3.8-27b", 24, "UD-IQ4_XS"),
    ("qwen-3.6-35b-a3b", 32, "UD-Q4_K_S"),
    ("qwen-3.6-35b-a3b", 48, "UD-Q5_K_XL"),  # capped at 5 bits below 64 GB
    ("qwen-3.6-35b-a3b", 64, "UD-Q8_K_XL"),
])
def test_recommended_quant_per_tier(group, ram, label):
    assert _star(group, ram) == label


def test_star_never_names_a_quant_that_does_not_fit_or_is_degraded():
    for ram in (16, 18, 24, 32, 36, 48, 64, 96, 128):
        for g in catalog.MODEL_GROUPS:
            rows = catalog.known_quants(g)
            i = catalog.recommended_quant_index(rows, g, ram)
            if i is None:
                continue
            filename, size = rows[i]
            assert catalog.quant_plan(g, size, ram)["fit"] == "fits", (g.key, ram, filename)
            assert not catalog.is_degraded(catalog._quant_label(filename), g), (g.key, ram, filename)


def test_drafter_dropped_only_to_reach_a_better_quality_tier():
    """Within a tier the drafter (1.5-2x decode) stays; dropping it is worth a
    jump from a 3-bit to a 4-bit quant."""
    drafter = catalog.Drafter("d.gguf", "r", "d.gguf", "draft-mtp", 1.0, 10 ** 9, "0" * 64)
    g = catalog.ModelGroup(key="t", display_name="Test 30B-A3B", maker="", hf_repo="t/t", family="",
                           architecture="qwen35moe", license="", notes="", drafter=drafter,
                           kv_bytes_per_token=10880)
    ws = gpu_memory.working_set_bytes(24) / 1e9
    room = ws - (gpu_memory.COMPUTE_RESERVE_BYTES + 10880 * 32768) / 1e9
    # Q4 fits only without the drafter; Q3 fits with it: take Q4.
    rows = [("t-UD-Q3_K_XL.gguf", room - 2.0), ("t-UD-Q4_K_XL.gguf", room - 0.5)]
    assert catalog.recommended_quant_index(rows, g, 24) == 1
    # Two 4-bit quants: keep the drafter with the smaller one.
    rows = [("t-UD-IQ4_XS.gguf", room - 2.0), ("t-UD-Q4_K_XL.gguf", room - 0.5)]
    assert catalog.recommended_quant_index(rows, g, 24) == 0
    assert catalog.quant_plan(g, room - 0.5, 24)["drafter"] is False
    assert catalog.quant_plan(g, room - 2.0, 24)["drafter"] is True


def test_picker_rows_label_degraded_quants_and_count_the_drafter_only_when_it_loads(tmp_path, monkeypatch):
    from localcode import hf_quants
    from localcode.ui import supervisor
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path))
    monkeypatch.setattr(hf_quants, "_read_cache", lambda _: None)
    monkeypatch.setattr(supervisor, "fetch_quants", lambda _: [])
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    sup.models_dir, sup.current, sup.ram_gb, sup.bandwidth = tmp_path, None, 16, 120
    sup.state, sup.active = {"state": "idle", "pct": None}, {}
    sup._quant_attempts = {"unsloth/Qwen3.8-27B-GGUF": float("inf"), "unsloth/gemma-4-12b-it-GGUF": float("inf")}
    rows = {q["filename"]: q for q in sup.quants("qwen-3.8-27b")["quants"]}
    iq2 = rows["Qwen3.8-27B-UD-IQ2_S.gguf"]
    assert iq2["label"] == "UD-IQ2_S · degraded" and iq2["degraded"] and iq2["fit"] == "fits"
    assert not any(q["recommended"] for q in rows.values())
    assert rows["Qwen3.8-27B-UD-Q2_K_XL.gguf"]["fit"] == "tight"
    gemma = {q["filename"]: q for q in sup.quants("gemma-4-12b")["quants"]}
    q5 = gemma["gemma-4-12b-it-UD-Q5_K_XL.gguf"]
    assert q5["recommended"] and q5["drafter"] and q5["size_gb"] == round(8.606021440 + 0.47, 1)
    q6 = gemma["gemma-4-12b-it-UD-Q6_K_XL.gguf"]
    assert q6["fit"] != "fits" and not q6["drafter"] and q6["size_gb"] == 10.7
