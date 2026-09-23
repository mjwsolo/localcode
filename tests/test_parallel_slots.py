"""Parallel slots trial: off by default, N from LOCALCODE_PARALLEL, RAM-guarded."""
from __future__ import annotations

import json
from pathlib import Path

from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway
from localcode.ui.launch import write_config


def _gw():
    return LocalCodeRuntimeGateway(RuntimeConfig())


def test_default_is_one_slot(monkeypatch):
    monkeypatch.delenv("LOCALCODE_PARALLEL", raising=False)
    assert _gw().parallel_slots(None) == 1


def test_env_requests_slots_and_ram_guard_caps_them(monkeypatch):
    monkeypatch.setenv("LOCALCODE_PARALLEL", "8")
    gw = _gw()
    # 64 GB, 16 GB weights, 131k ctx at 32 KB/token = 4 GB per slot -> budget
    # after reserve (~9.6 GB) is ~38 GB -> 9 slots fit, so the request of 8 stands.
    monkeypatch.setattr(gw, "_system_ram_gb", lambda: 64)
    monkeypatch.setattr(gw, "_model_file_bytes", lambda p: 16 * 1024 ** 3)
    monkeypatch.setattr(gw, "_kv_bytes_per_token", lambda p: 32 * 1024)
    monkeypatch.setattr(gw, "_target_num_ctx", lambda model_path=None, **k: 131072)
    assert gw.parallel_slots("x.gguf") == 8
    # 24 GB: budget after reserve is ~4.4 GB -> 1 slot.
    monkeypatch.setattr(gw, "_system_ram_gb", lambda: 24)
    assert gw.parallel_slots("x.gguf") == 1


def test_env_unknown_kv_trusts_request_and_caps_at_16(monkeypatch):
    monkeypatch.setenv("LOCALCODE_PARALLEL", "40")
    gw = _gw()
    monkeypatch.setattr(gw, "_kv_bytes_per_token", lambda p: None)
    assert gw.parallel_slots("x.gguf") == 16


def test_config_enables_task_tool_only_with_slots(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALCODE_PARALLEL", raising=False)
    write_config(tmp_path / "a.json", port=8123, ctx=1000, alias=None)
    a = json.loads((tmp_path / "a.json").read_text())
    assert a["tools"] == {"task": False} and "subagent_depth" not in a
    monkeypatch.setenv("LOCALCODE_PARALLEL", "4")
    write_config(tmp_path / "b.json", port=8123, ctx=1000, alias=None)
    b = json.loads((tmp_path / "b.json").read_text())
    assert b["tools"] == {"task": True} and b["subagent_depth"] == 1
