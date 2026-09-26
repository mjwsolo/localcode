"""Parallel slots: N from the RAM tier by default, LOCALCODE_PARALLEL overrides, RAM-guarded."""
from __future__ import annotations

import json

import pytest

from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway
from localcode.ui import launch
from localcode.ui.launch import write_config


def _gw(ram_gb: int = 128):
    gw = LocalCodeRuntimeGateway(RuntimeConfig())
    gw._system_ram_gb = lambda: ram_gb  # type: ignore[method-assign]
    return gw


@pytest.mark.parametrize("ram,expected", [(16, 1), (32, 1), (48, 1), (64, 2), (96, 3), (128, 4), (192, 4)])
def test_default_slots_follow_the_ram_tier(monkeypatch, ram, expected):
    monkeypatch.delenv("LOCALCODE_PARALLEL", raising=False)
    gw = _gw(ram)
    monkeypatch.setattr(gw, "_kv_bytes_per_token", lambda p: None)
    assert gw.parallel_slots(None) == expected


def test_env_overrides_the_tier_and_one_turns_it_off(monkeypatch):
    gw = _gw(128)
    monkeypatch.setattr(gw, "_kv_bytes_per_token", lambda p: None)
    monkeypatch.setenv("LOCALCODE_PARALLEL", "1")
    assert gw.parallel_slots(None) == 1
    monkeypatch.setenv("LOCALCODE_PARALLEL", "6")
    assert gw.parallel_slots(None) == 6
    monkeypatch.setenv("LOCALCODE_PARALLEL", "junk")
    assert gw.parallel_slots(None) == 4


def test_ram_guard_caps_slots(monkeypatch):
    monkeypatch.setenv("LOCALCODE_PARALLEL", "8")
    gw = _gw(64)
    # 64 GB, 16 GB weights, 131k ctx at 32 KB/token = 4 GB per slot -> budget
    # after reserve (~9.6 GB) is ~38 GB -> 9 slots fit, so the request of 8 stands.
    monkeypatch.setattr(gw, "_model_file_bytes", lambda p: 16 * 1024 ** 3)
    monkeypatch.setattr(gw, "_kv_bytes_per_token", lambda p: 32 * 1024)
    monkeypatch.setattr(gw, "_target_num_ctx", lambda model_path=None, **k: 131072)
    assert gw.parallel_slots("x.gguf") == 8
    # 24 GB: budget after reserve is ~4.4 GB -> 1 slot.
    gw._system_ram_gb = lambda: 24  # type: ignore[method-assign]
    assert gw.parallel_slots("x.gguf") == 1


def test_env_unknown_kv_trusts_request_and_caps_at_16(monkeypatch):
    monkeypatch.setenv("LOCALCODE_PARALLEL", "40")
    gw = _gw(128)
    monkeypatch.setattr(gw, "_kv_bytes_per_token", lambda p: None)
    assert gw.parallel_slots("x.gguf") == 16


def test_config_enables_task_tool_only_with_slots(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "parallel_slots", lambda: 1)
    write_config(tmp_path / "a.json", port=8123, ctx=1000, alias=None)
    a = json.loads((tmp_path / "a.json").read_text())
    assert a["tools"] == {"task": False} and "subagent_depth" not in a
    monkeypatch.setattr(launch, "parallel_slots", lambda: 4)
    write_config(tmp_path / "b.json", port=8123, ctx=1000, alias=None)
    b = json.loads((tmp_path / "b.json").read_text())
    assert b["tools"] == {"task": True} and b["subagent_depth"] == 1
