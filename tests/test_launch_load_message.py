"""Launching with --model says something while the model loads."""
from types import SimpleNamespace

from localcode.ui import launch


def test_model_load_wait_is_announced(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "ui_binary_path", lambda: tmp_path / "ui")
    monkeypatch.setattr(launch, "_llama_server", lambda: tmp_path / "server")
    monkeypatch.setattr(launch, "_models_dir", lambda: tmp_path)
    monkeypatch.setattr(launch, "_model_alias", lambda m, d: "wanted")
    monkeypatch.setattr(launch, "find_running", lambda: None)
    monkeypatch.setattr(launch, "choose_ports", lambda: (1, 2))
    monkeypatch.setattr(launch, "run_dir", lambda: tmp_path / "run")
    monkeypatch.setattr(launch, "WAIT_S", -1)
    monkeypatch.setattr(launch, "_ok", lambda url: False)
    monkeypatch.setattr(launch.signal, "signal", lambda *a: None)
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: SimpleNamespace(
        poll=lambda: None, terminate=lambda: None, wait=lambda timeout=None: 0, kill=lambda: None))
    assert launch.main("wanted") == 1
    assert "loading wanted" in capsys.readouterr().err
