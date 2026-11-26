"""The one fit rule: files plus measured run-time memory inside the GPU working set."""
import pytest

from localcode import fit as F


@pytest.fixture(autouse=True)
def emulate(monkeypatch):
    # Use the documented fractions, never the host Mac's Metal value.
    monkeypatch.setenv("LOCALCODE_RAM_GB", "16")


def test_gpu_budget_fractions():
    assert F.gpu_budget_gb(16) == pytest.approx(10.667, abs=0.01)
    assert F.gpu_budget_gb(36) == pytest.approx(24.0)
    assert F.gpu_budget_gb(64) == pytest.approx(48.0)


@pytest.mark.parametrize("name,files,ram,want", [
    ("gemma-4-12b-it-UD-Q4_K_XL.gguf", 7.0, 16, "fits"),      # 9.2 GB measured
    ("Qwen3.8-27B-UD-IQ2_S.gguf", 8.7, 16, "too_big"),        # 13.0 GB measured
    ("Qwen3.8-27B-UD-IQ2_S.gguf", 8.7, 24, "fits"),
    ("Qwen3.6-35B-A3B-UD-IQ2_M.gguf", 11.6, 24, "fits"),      # 13.9 GB measured
    ("Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf", 39.7, 48, "too_big"),
    ("Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf", 39.7, 64, "fits"),
    # DeepSeek V4 Flash Q2_K_XL + projector: 2.0 GB measured beyond the files.
    ("DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003.gguf", 97.8, 96, "too_big"),
])
def test_measured_models(name, files, ram, want):
    assert F.fit(files, ram, name) == want


def test_unmeasured_models_use_a_conservative_default():
    assert F.runtime_overhead_gb("Some-Future-Model-Q4.gguf", 10.0) == pytest.approx(6.0)


def test_emulated_128_gb_mac_uses_metal_measured_working_set(monkeypatch):
    monkeypatch.setenv("LOCALCODE_RAM_GB", "128")
    assert F.gpu_budget_gb(128) == pytest.approx(107.52)
    assert F.gpu_budget_gb(96) == pytest.approx(72.0)


def test_deepseek_v4_flash_fits_a_128_gb_mac(monkeypatch):
    monkeypatch.setenv("LOCALCODE_RAM_GB", "128")
    name = "DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003.gguf"
    assert F.fit(96.83 + 0.93, 128, name) == "fits"
    assert F.fit(114.39 + 0.93, 128, name) == "too_big"   # UD-IQ3_S


# ---- streamed experts -------------------------------------------------------
from localcode.gguf_experts import MoeInfo  # noqa: E402

# 26 GB of files, 24 GB of them experts: 256 experts, 8 used, 40 layers.
MOE = MoeInfo(expert_bytes=int(24e9), n_expert=256, n_expert_used=8, n_moe_layers=40)
QWEN = "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"


def test_moe_that_misses_the_gpu_but_fits_ram_streams(monkeypatch):
    monkeypatch.setenv("LOCALCODE_RAM_GB", "36")
    assert F.fit(26.0, 36, QWEN) == "too_big"
    assert F.fit(26.0, 36, QWEN, MOE) == "streamed"


def test_moe_that_fits_the_gpu_is_unchanged(monkeypatch):
    monkeypatch.setenv("LOCALCODE_RAM_GB", "64")
    assert F.fit(39.7, 64, QWEN, MOE) == "fits"


def test_dense_or_unknown_model_never_streams(monkeypatch):
    monkeypatch.setenv("LOCALCODE_RAM_GB", "36")
    assert F.fit(26.0, 36, QWEN, None) == "too_big"


def test_files_beyond_ram_do_not_stream(monkeypatch):
    monkeypatch.setenv("LOCALCODE_RAM_GB", "36")
    # 36 - max(3, 5.4) = 30.6 GB of RAM for files + run-time memory; 29 + 2.35 > 30.6
    assert F.fit(29.0, 36, QWEN, MOE) == "too_big"


def test_cache_size_is_what_the_gpu_has_left(monkeypatch):
    monkeypatch.setenv("LOCALCODE_RAM_GB", "36")
    got = F.moe_cache_mib(26.0, 36, QWEN, MOE)
    assert got == int((24.0 - 0.5 - 2.0 - 2.35) * 1024)


def test_cache_below_the_smallest_bank_is_too_big(monkeypatch):
    monkeypatch.setenv("LOCALCODE_RAM_GB", "36")
    huge_bank = MoeInfo(int(24e9), n_expert=8, n_expert_used=8, n_moe_layers=2)  # one token touches every expert
    assert huge_bank.min_bank_bytes == 24e9
    assert F.moe_cache_mib(26.0, 36, QWEN, huge_bank) is None
    assert F.fit(26.0, 36, QWEN, huge_bank) == "too_big"
