"""--model on a second launch must say so when the running service has no model."""
from localcode.ui import launch


def test_attach_to_idle_service_reports_ignored_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path))
    monkeypatch.setattr(launch, "read_auth_file", lambda _: None)
    monkeypatch.setattr(launch, "run_dir", lambda: tmp_path)
    monkeypatch.setattr(launch, "runtime_env", lambda *a, **k: {})
    monkeypatch.setattr(launch.subprocess, "call", lambda *a, **k: 0)
    status = {"port": 8123, "inference_port": 8123, "ctx": 4096, "current": ""}
    assert launch.attach(8323, status, tmp_path / "ui", tmp_path, "wanted") == 0
    assert "wanted" in capsys.readouterr().err
