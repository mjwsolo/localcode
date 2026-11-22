"""A dictation worker that never confirms the microphone is not reported as recording."""
import sys

from localcode.ui import supervisor as sv


def test_unconfirmed_worker_is_killed_and_reported(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "voice_live.py").write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(sv, "VOICE_OPEN_WAIT_S", 0.3)
    monkeypatch.setattr(sv.localcode, "__file__", str(pkg / "__init__.py"))
    monkeypatch.setattr("localcode.voice.VoiceState", lambda: type("V", (), {"stt_model_path": tmp_path / "m"})())
    sup = sv.Supervisor.__new__(sv.Supervisor)
    sup.control_port, sup.control_token = 1, "t"
    result = sup._voice_start_live(sys.executable)
    assert "error" in result and "did not start" in result["error"]
    assert sup.voice_proc is None and sup.voice_live is False
