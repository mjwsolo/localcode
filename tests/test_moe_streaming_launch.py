"""A MoE quant that misses the GPU working set but fits RAM launches with a
streamed expert cache; models that fit are launched exactly as before."""
from gguf_builder import moe_tensors, write_gguf

from localcode.config import AppConfig, RuntimeConfig, UIConfig
from localcode.runtime import LocalCodeRuntimeGateway
from localcode.ui import server_cmd

KV = {"qwen3moe.expert_count": 256, "qwen3moe.expert_used_count": 8, "qwen3moe.block_count": 40}
NAME = "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"


def _setup(monkeypatch, tmp_path, ram, expert_layer_bytes):
    gguf = write_gguf(tmp_path / NAME, moe_tensors(40, expert_layer_bytes, dense_bytes=int(0.05e9)), KV)
    cfg = AppConfig(runtime=RuntimeConfig(), ui=UIConfig())
    monkeypatch.setattr(server_cmd, "load_config", lambda: cfg)
    monkeypatch.setattr(server_cmd, "mmproj_for", lambda g: None)
    monkeypatch.setattr(LocalCodeRuntimeGateway, "_system_ram_gb", lambda _: ram)
    monkeypatch.setenv("LOCALCODE_RAM_GB", str(ram))
    monkeypatch.setenv("LOCALCODE_PROMPT_WARMUP", "0")
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path / "run"))
    return gguf


def test_streamed_model_gets_cache_flags(monkeypatch, tmp_path):
    # 40 layers x 3 tensors x 200 MB = 24 GB of experts, plus ~2 GB of other tensors.
    gguf = _setup(monkeypatch, tmp_path, 36, 200_000_000)
    cmd = server_cmd.server_command(str(gguf), 8123)
    assert cmd[cmd.index("--load-mode") + 1] == "mmap"
    assert cmd[cmd.index("-ot") + 1] == "exps=CPU"
    assert cmd.count("-ot") == 1
    mib = int(cmd[cmd.index("--moe-cache-mib") + 1])
    assert 10_000 < mib < 24_000


def test_model_that_fits_is_launched_unchanged(monkeypatch, tmp_path):
    gguf = _setup(monkeypatch, tmp_path, 64, 200_000_000)
    cmd = server_cmd.server_command(str(gguf), 8123)
    assert "--moe-cache-mib" not in cmd


def test_model_beyond_ram_is_still_refused(monkeypatch, tmp_path):
    import pytest
    gguf = _setup(monkeypatch, tmp_path, 24, 200_000_000)
    with pytest.raises(ValueError):
        server_cmd.server_command(str(gguf), 8123)
