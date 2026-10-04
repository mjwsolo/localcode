"""The default interface's Python side: launcher config, frontend selection,
binary resolution. No model, no binary execution."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from localcode import entrypoint
from localcode.ui import fork_commit, plugin_path, run_dir, ui_binary_path
from localcode.ui.launch import write_config


def test_parser_accepts_version_model_and_project():
    p = entrypoint.build_parser()
    a = p.parse_args(["--version"])
    assert a.version is True
    a = p.parse_args(["--model", "x", "somedir"])
    assert a.model == "x" and a.project == "somedir"
    for flag in ("-s", "--session", "--resume"):
        assert p.parse_args([flag, "ses_example"]).resume == "ses_example"
    with pytest.raises(SystemExit):
        p.parse_args(["--classic"])  # the previous interface is gone


def test_unsupported_platform_refuses_plainly(monkeypatch, capsys):
    monkeypatch.setattr("localcode.ui.platform_supported", lambda: False)
    with pytest.raises(SystemExit) as e:
        entrypoint.main([])
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "Apple silicon" in err and "did not work there either" in err


def test_missing_binary_refuses_plainly(monkeypatch, capsys):
    monkeypatch.setattr("localcode.ui.platform_supported", lambda: True)
    monkeypatch.setattr("localcode.ui.ui_binary_path", lambda: None)
    with pytest.raises(SystemExit) as e:
        entrypoint.main([])
    assert e.value.code == 1
    assert "pip install -U localcode" in capsys.readouterr().err


def test_missing_project_dir_is_an_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        entrypoint.main([str(tmp_path / "nope")])
    assert e.value.code == 1
    assert "directory not found" in capsys.readouterr().err


def test_entrypoint_forwards_session_to_launcher(monkeypatch):
    from localcode.ui import launch
    monkeypatch.setattr("localcode.ui.platform_supported", lambda: True)
    monkeypatch.setattr("localcode.ui.ui_binary_path", lambda: Path("/fake/ui"))
    calls = []
    monkeypatch.setattr(launch, "main", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    with pytest.raises(SystemExit) as exc:
        entrypoint.main(["-s", "ses_example"])
    assert exc.value.code == 0
    assert calls == [((None,), {"project": None, "resume": "ses_example"})]


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



def test_second_launch_attaches_to_running_session(monkeypatch, tmp_path):
    """Another terminal reuses the running session's model server instead of being refused."""
    import json as _json
    from localcode.ui import launch
    calls = {}
    monkeypatch.setattr(launch, "ui_binary_path", lambda: tmp_path / "ui")
    (tmp_path / "ui").write_bytes(b"x")
    monkeypatch.setattr(launch, "_llama_server", lambda: tmp_path / "ui")
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path / "run"))
    status = {"port": 8123, "ctx": 65536, "current": "gemma-4-12b-it-UD-Q4_K_XL", "state": "ready"}
    monkeypatch.setattr(launch, "find_running", lambda: (8323, status))
    def fake_call(argv, cwd, env):
        calls["argv"] = argv; calls["env"] = env
        cfg = _json.loads(Path(env["LOCALCODE_CONFIG"]).read_text())
        calls["cfg"] = cfg
        return 0
    monkeypatch.setattr(launch.subprocess, "call", fake_call)
    assert launch.main(None, str(tmp_path)) == 0
    assert calls["env"]["LOCALCODE_CONTROL_URL"] == "http://127.0.0.1:8323"
    assert calls["argv"][1:] == ["-m", "localcode/gemma-4-12b-it-UD-Q4_K_XL"]
    assert calls["cfg"]["provider"]["localcode"]["options"]["baseURL"] == "http://127.0.0.1:8123/v1"
    assert calls["cfg"]["model"] == "localcode/gemma-4-12b-it-UD-Q4_K_XL"


def test_resume_flags_reach_attached_runtime(monkeypatch, tmp_path):
    from localcode.ui import launch
    binary = tmp_path / "ui"
    binary.write_bytes(b"x")
    monkeypatch.setattr(launch, "ui_binary_path", lambda: binary)
    monkeypatch.setattr(launch, "_llama_server", lambda: binary)
    monkeypatch.setattr(launch, "find_running", lambda: (8323, {"port": 8123, "ctx": 8192}))
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path / "run"))
    calls = []
    monkeypatch.setattr(launch.subprocess, "call", lambda argv, **kwargs: calls.append((argv, kwargs)) or 0)
    assert launch.main(project=str(tmp_path), resume="ses_example") == 0
    assert calls[-1][0][-2:] == ["--session", "ses_example"]
    assert launch.main(project=str(tmp_path), resume="last") == 0
    assert calls[-1][0][-1:] == ["--continue"]


def test_session_id_uses_its_saved_project(monkeypatch, tmp_path):
    from localcode.ui.launch import _session_directory
    data = tmp_path / "data" / "localcode-agent"
    data.mkdir(parents=True)
    project = tmp_path / "original-project"
    project.mkdir()
    with sqlite3.connect(data / "opencode.db") as conn:
        conn.execute("CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT)")
        conn.execute("INSERT INTO session VALUES (?, ?)", ("ses_example", str(project)))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert _session_directory("ses_example") == project
    assert _session_directory("ses_missing") is None
