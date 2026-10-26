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
])
def test_measured_models(name, files, ram, want):
    assert F.fit(files, ram, name) == want


def test_unmeasured_models_use_a_conservative_default():
    assert F.runtime_overhead_gb("Some-Future-Model-Q4.gguf", 10.0) == pytest.approx(6.0)
