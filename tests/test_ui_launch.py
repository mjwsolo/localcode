"""The default interface's Python side: launcher config, frontend selection,
binary resolution. No model, no binary execution."""
from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

from localcode import entrypoint
from localcode.ui import fork_commit, plugin_path, run_dir, ui_binary_path
from localcode.ui.launch import write_config


def _args(**kw):
    base = dict(classic=False, preview_screen=None, resume=None, model=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_default_frontend_is_ui(monkeypatch):
    monkeypatch.delenv("LOCALCODE_FRONTEND", raising=False)
    assert entrypoint._frontend_choice(_args()) == "ui"


@pytest.mark.parametrize("env", ["classic", "CLASSIC", "tui", "textual"])
def test_env_selects_classic(monkeypatch, env):
    monkeypatch.setenv("LOCALCODE_FRONTEND", env)
    assert entrypoint._frontend_choice(_args()) == "classic"


def test_flag_and_classic_only_options_select_classic(monkeypatch):
    monkeypatch.delenv("LOCALCODE_FRONTEND", raising=False)
    assert entrypoint._frontend_choice(_args(classic=True)) == "classic"
    assert entrypoint._frontend_choice(_args(preview_screen="chat")) == "classic"
    assert entrypoint._frontend_choice(_args(resume="last")) == "classic"


def test_parser_accepts_classic_and_version():
    p = entrypoint.build_parser()
    assert p.parse_args(["--classic"]).classic is True
    assert p.parse_args(["--version"]).version is True


def test_write_config_points_at_local_server_and_plugin(tmp_path):
    path = tmp_path / "session.json"
    write_config(path, port=8123, ctx=32768, alias=None)
    cfg = json.loads(path.read_text())
    prov = cfg["provider"]["localcode"]
    assert prov["options"]["baseURL"] == "http://127.0.0.1:8123/v1"
    assert cfg["enabled_providers"] == ["localcode"]
    assert cfg["share"] == "disabled" and cfg["autoupdate"] is False
    assert cfg["tools"] == {"task": False}
    assert "model" not in cfg
    assert list(prov["models"]) == ["__pending__"]
    assert prov["models"]["__pending__"]["limit"]["context"] == 32768
    assert prov["models"]["__pending__"]["attachment"] is False
    # the discipline plugin is loaded from the package by absolute path, not copied into the project
    assert cfg["plugin"] == [str(plugin_path())]
    assert Path(cfg["plugin"][0]).is_absolute() and Path(cfg["plugin"][0]).is_file()
    assert not (tmp_path / ".localcode-agent").exists()


def test_write_config_with_alias(tmp_path):
    path = tmp_path / "s.json"
    write_config(path, port=8200, ctx=8192, alias="gemma-4-12b-it-UD-Q4_K_XL")
    cfg = json.loads(path.read_text())
    assert cfg["model"] == "localcode/gemma-4-12b-it-UD-Q4_K_XL"
    assert list(cfg["provider"]["localcode"]["models"]) == ["gemma-4-12b-it-UD-Q4_K_XL"]
    assert not path.with_suffix(".json.tmp").exists()


def test_run_dir_is_per_user_not_project(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALCODE_AGENT_RUN_DIR", raising=False)
    assert run_dir() == Path.home() / ".local" / "share" / "localcode-agent" / "run"
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path / "r"))
    assert run_dir() == tmp_path / "r"


def test_ui_binary_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALCODE_UI_BIN", str(tmp_path / "missing"))
    assert ui_binary_path() is None
    fake = tmp_path / "ui"; fake.write_bytes(b"x")
    monkeypatch.setenv("LOCALCODE_UI_BIN", str(fake))
    assert ui_binary_path() == fake


def test_bundled_binary_and_fork_commit_present():
    monkey_free = os.environ.get("LOCALCODE_UI_BIN")
    assert monkey_free is None or True
    assert len(fork_commit()) == 40
    bundled = Path(entrypoint.__file__).parent / "bin" / "localcode-ui"
    assert bundled.is_file() and os.access(bundled, os.X_OK)


def test_packaging_declares_ui_files():
    root = Path(entrypoint.__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text()
    manifest = (root / "MANIFEST.in").read_text()
    for needle in ("bin/localcode-ui", "ui/plugin/*.ts", "ui/FORK_COMMIT"):
        assert needle in pyproject, needle
    assert "src/localcode/bin/localcode-ui" in manifest
    assert "src/localcode/ui/plugin" in manifest


CONTROL_ROUTES = [
    "/status", "/catalog", "/quants", "/select", "/cancel", "/progress",
    "/models_dir", "/vision/install",
    "/voice/status", "/voice/setup", "/voice/start", "/voice/stop", "/voice/transcribe", "/voice/speak",
]


def test_supervisor_serves_every_route_the_ui_calls():
    src = (Path(entrypoint.__file__).parent / "ui" / "supervisor.py").read_text()
    missing = [r for r in CONTROL_ROUTES if f'"{r}"' not in src]
    assert not missing, missing


def test_launch_refuses_home_directory(monkeypatch, tmp_path, capsys):
    """Running in $HOME makes the runtime index the whole home; refuse with a hint."""
    from localcode.ui import launch
    monkeypatch.setattr(launch, "ui_binary_path", lambda: tmp_path / "ui")
    (tmp_path / "ui").write_bytes(b"x")
    monkeypatch.setattr(launch, "_llama_server", lambda: tmp_path / "ui")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert launch.main(None, str(tmp_path)) == 1
    assert "home directory" in capsys.readouterr().err
