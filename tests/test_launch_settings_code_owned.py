"""Launch tuning is worked out for the Mac at every launch, never taken from config.toml.

A saved value froze whatever it was when written, across upgrades, and every
fresh install wrote a hard-coded thread count.
"""
from pathlib import Path

from localcode import config as cfgmod
from localcode.config import load_config, save_config
from localcode.runtime import LocalCodeRuntimeGateway

STALE = """[runtime]
provider = "llama_cpp"
laptop_26b_runtime_mode = "context"
llama_cpp_gpu_layers = 0
llama_cpp_threads = 4
llama_cpp_batch_size = 64
llama_cpp_draft_max = 3
llama_cpp_mtp_draft = 0
llama_cpp_expert_offload = true
llama_cpp_draft_model = "/old/drafter.gguf"
kv_cache_type_k = "turbo4"
kv_cache_type_v = "turbo2"
llama_cpp_cache_reuse = 0
temperature = 0.7
"""

LAUNCH_ENV = [k for k in __import__("os").environ if k.startswith("LOCALCODE_LLAMA_CPP_") or k.startswith("LOCALCODE_KV_CACHE")
              or k == "LOCALCODE_LAPTOP_26B_RUNTIME_MODE"]


def _command(monkeypatch, tmp_path, text):
    path = tmp_path / "config.toml"
    if text is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(text)
    monkeypatch.setattr(cfgmod, "get_config_path", lambda: path)
    for k in LAUNCH_ENV:
        monkeypatch.delenv(k, raising=False)
    cfg = load_config()
    gw = LocalCodeRuntimeGateway(cfg.runtime)
    monkeypatch.setattr(gw, "_system_ram_gb", lambda: 64)
    return cfg, gw.llama_server_command(str(tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf"), 8123)


def test_a_stale_config_cannot_change_the_launch_command(monkeypatch, tmp_path):
    _, fresh = _command(monkeypatch, tmp_path, None)
    cfg, stale = _command(monkeypatch, tmp_path, STALE)
    assert stale == fresh
    assert "--threads" in fresh and fresh[fresh.index("--threads") + 1] != "4"
    # Settings the user does own still load from the file.
    assert cfg.runtime.temperature == 0.7


def test_saving_never_writes_launch_tuning(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(STALE)
    monkeypatch.setattr(cfgmod, "get_config_path", lambda: path)
    save_config(load_config())
    text = path.read_text()
    for key in ("llama_cpp_threads", "llama_cpp_batch_size", "llama_cpp_gpu_layers", "kv_cache_type_k",
                "kv_cache_type_v", "laptop_26b_runtime_mode", "llama_cpp_mtp_draft", "llama_cpp_draft_model"):
        assert f"{key} =" not in text, key
    assert "temperature = 0.7" in text
    assert "llama_cpp_threads" not in cfgmod.DEFAULT_CONFIG


def test_threads_follow_the_performance_cores(monkeypatch, tmp_path):
    import platform
    import localcode.runtime as rt
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(rt.os if hasattr(rt, "os") else __import__("os"), "cpu_count", lambda: 18)
    for perf, want in ((12, "12"), (4, "4"), (6, "6"), (24, "12"), (None, "12")):
        monkeypatch.setattr(rt, "_performance_cores", lambda p=perf: p)
        _, cmd = _command(monkeypatch, tmp_path, None)
        assert cmd[cmd.index("--threads") + 1] == want, (perf, cmd[cmd.index("--threads") + 1])
