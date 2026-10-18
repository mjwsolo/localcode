"""Lazy image projector and the memory-tight drafter rule.

The server starts text-only; the first image asks the supervisor to restart
it with the projector already on disk (never a download). The catalog drafter
loads only when it fits beside the model in the GPU working set.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from localcode import models_catalog
from localcode.config import AppConfig, RuntimeConfig, UIConfig
from localcode.runtime import LocalCodeRuntimeGateway
from localcode.ui import server_cmd, supervisor

QWEN = models_catalog.by_key("qwen")


def _sparse(path: Path, size: int) -> Path:
    with path.open("wb") as f:
        f.truncate(size)
    return path


@pytest.fixture
def shipping(monkeypatch, tmp_path):
    monkeypatch.setattr(models_catalog, "model_dir", lambda: tmp_path)
    config = AppConfig(runtime=RuntimeConfig(), ui=UIConfig())
    config.runtime.llama_cpp_binary = str(tmp_path / "llama-server")
    Path(config.runtime.llama_cpp_binary).touch()
    monkeypatch.setattr(server_cmd, "load_config", lambda: config)
    monkeypatch.setenv("LOCALCODE_SERVER_KEY", "test-only-key")
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("LOCALCODE_PROMPT_WARMUP", "0")

    def at(ram):
        monkeypatch.setattr(LocalCodeRuntimeGateway, "_system_ram_gb", lambda _: ram)
        monkeypatch.setenv("LOCALCODE_RAM_GB", str(ram))
    return tmp_path, at


def test_server_starts_without_the_projector_and_adds_it_on_request(shipping):
    tmp_path, at = shipping
    at(128)
    model = _sparse(tmp_path / QWEN.filename, QWEN.size_bytes)
    _sparse(tmp_path / QWEN.mmproj_filename, 899_283_680)
    text = server_cmd.server_command(str(model), 8123, "qwen")
    assert "--mmproj" not in text
    seeing = server_cmd.server_command(str(model), 8123, "qwen", vision=True)
    assert seeing[seeing.index("--mmproj") + 1] == str(tmp_path / QWEN.mmproj_filename)


def test_drafter_kept_where_it_fits_and_dropped_when_memory_is_tight(shipping):
    tmp_path, at = shipping
    drafter = QWEN.drafter
    from test_gpu_memory import _gguf
    head = _gguf(tmp_path / drafter.filename, "dflash", {
        "block_count": 6, "attention.head_count": 32, "attention.head_count_kv": 4,
        "attention.key_length": 128, "attention.value_length": 128})
    with head.open("r+b") as f:
        f.truncate(drafter.size_bytes)
    # Qwen 3.6 UD-IQ3_S (13.68 GB): with the 0.42 GB DFlash drafter it fits a
    # 24 GB Mac's 16 GiB working set beside a working context.
    def qwen(name, size):
        path = _gguf(tmp_path / name, "qwen35moe", {
            "block_count": 40, "attention.head_count": 16, "attention.head_count_kv": 2,
            "attention.key_length": 256, "attention.value_length": 256, "full_attention_interval": 4})
        with path.open("r+b") as f:
            f.truncate(size)
        return path
    iq3 = qwen("Qwen3.6-35B-A3B-UD-IQ3_S.gguf", 13_676_723_168)
    at(24)
    cmd = server_cmd.server_command(str(iq3), 8123)
    assert cmd[cmd.index("--model-draft") + 1] == str(tmp_path / drafter.filename)
    # UD-Q3_K_S (15.36 GB) only fits without it: the drafter is left out.
    q3ks = qwen("Qwen3.6-35B-A3B-UD-Q3_K_S.gguf", 15_359_196_128)
    cmd = server_cmd.server_command(str(q3ks), 8123)
    assert "--model-draft" not in cmd and "--spec-type" not in cmd
    # The picker shows the same decision.
    group = models_catalog.by_group("qwen-3.6-35b-a3b")
    assert models_catalog.quant_plan(group, 13.676723168, 24)["drafter"] is True
    assert models_catalog.quant_plan(group, 15.359196128, 24)["drafter"] is False


def _sup(tmp_path, monkeypatch, on_disk=True):
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    sup.models_dir = tmp_path
    sup.current = "Qwen3.6-35B-A3B-UD-IQ2_M"
    sup.lock = threading.Lock()
    sup.state = {"state": "ready", "model": sup.current, "detail": "", "pct": None}
    sup.active = {}
    sup.vision_wanted, sup.vision_loaded = None, False
    sup.progress = lambda: {"phase": "idle"}
    sup.cancel_warmup = lambda: {"ok": True}
    started = []

    def start(alias):
        started.append((alias, sup.vision_wanted))
        sup.vision_loaded = sup.vision_wanted == alias
        return True
    sup.start = start
    projector = tmp_path / QWEN.mmproj_filename
    monkeypatch.setattr(server_cmd, "mmproj_for", lambda _: projector if on_disk else None)
    return sup, started


def test_first_image_restarts_the_server_with_the_projector(tmp_path, monkeypatch):
    sup, started = _sup(tmp_path, monkeypatch)
    info = sup.vision_info()
    assert info["vision"] is True and info["vision_loaded"] is False  # images accepted, not loaded
    res = sup.vision_load()
    assert res == {"ok": True, "loading": True, "size_gb": 0.9}
    for _ in range(100):
        if sup.state["state"] == "ready" and started:
            break
        time.sleep(0.02)
    assert started == [(sup.current, sup.current)]
    assert sup.vision_info()["vision_loaded"] is True
    assert sup.vision_load() == {"ok": True, "loaded": True}  # once per load
    assert len(started) == 1


def test_missing_projector_is_never_downloaded_by_an_image(tmp_path, monkeypatch):
    sup, started = _sup(tmp_path, monkeypatch, on_disk=False)
    from localcode import bootstrap
    monkeypatch.setattr(bootstrap, "download_mmproj", lambda *a, **k: pytest.fail("downloaded"))
    res = sup.vision_load()
    assert "/vision" in res["error"] and res["vision_available"] is True
    assert not started


def test_start_passes_the_projector_request_to_the_server_command(tmp_path, monkeypatch):
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    sup._closing = threading.Event()
    sup.models_dir, sup.port = tmp_path, 8123
    seen = []

    def command(gguf, port, alias, vision=False):
        seen.append(vision)
        raise ValueError("stop before spawning")
    monkeypatch.setattr(server_cmd, "server_command", command)
    sup.vision_wanted = None
    with pytest.raises(ValueError):
        sup.start("model-a")
    sup.vision_wanted = "model-a"
    with pytest.raises(ValueError):
        sup.start("model-a")
    with pytest.raises(ValueError):
        sup.start("model-b")
    assert seen == [False, True, False]


def test_switching_models_starts_the_new_one_text_only(tmp_path):
    (tmp_path / "other.gguf").write_bytes(b"GGUF")
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    sup.models_dir, sup.current, sup.proc = tmp_path, "previous", None
    sup.lock = threading.Lock(); sup.lock.acquire()
    sup.cancel = threading.Event(); sup.active = {}
    sup.state = {"state": "loading", "model": "other", "detail": "", "pct": None}
    sup.vision_wanted = "previous"
    wanted = []
    sup.start = lambda alias: wanted.append(sup.vision_wanted) or True
    sup._select_worker(type("G", (), {"drafter": None})(), "other.gguf", "other")
    assert wanted == [None]
