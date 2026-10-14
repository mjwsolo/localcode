"""Live dictation: the pacing rules, and the worker against a spoken clip.

The clip tests need the voice runtime and speech model that `/voice` installs;
without them they are skipped. Nothing is downloaded.
"""
import json
import os
import subprocess
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src/localcode/voice_live.py"
VOICE_PY = Path.home() / ".local/share/localcode-agent/run/voice-venv/bin/python"
MODEL = Path.home() / ".local/share/localcode/voice/ggml-medium.en-q5_0.bin"


def test_pacing_helpers():
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(ROOT / "src"))
    from localcode import voice_live as v

    rate = v.RATE
    t = np.arange(rate) / rate
    speech = (np.sin(2 * np.pi * 220 * t) * 6000).astype("int16")          # one loud second
    silence = np.zeros(rate, dtype="int16")
    assert v.has_speech(speech) and not v.has_speech(silence)
    assert v.trailing_silence(np.concatenate([speech, silence])) == pytest.approx(1.0, abs=0.05)
    assert v.trailing_silence(speech) == 0
    # an over-long tail is cut inside its quiet gap, not in the middle of a word
    tail = np.concatenate([speech] * 9 + [silence[: rate // 2]] + [speech] * 3)
    cut = v.quietest_cut(tail)
    assert 9 * rate <= cut <= 9.5 * rate
    assert v.join("one two", " three ") == "one two three" and v.join("", "x") == "x" and v.join("x", "") == "x"


def _clip(tmp_path) -> Path:
    if sys.platform != "darwin" or not VOICE_PY.exists() or not MODEL.exists():
        pytest.skip("needs the voice runtime and speech model installed by /voice")
    aiff, wav = tmp_path / "clip.aiff", tmp_path / "clip.wav"
    subprocess.run(["say", "-o", str(aiff), "Open the file called calculator. [[slnc 1000]] Then run the test suite."], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
    with wave.open(str(wav)) as w:
        assert w.getframerate() == 16000
    return wav


def _run(wav: Path, env: dict) -> list[dict]:
    r = subprocess.run([str(VOICE_PY), str(WORKER), str(MODEL), "--wav", str(wav)], input="", capture_output=True, text=True, timeout=180,
                       env={**os.environ, **env})
    assert r.returncode == 0, f"voice worker exited {r.returncode}: {r.stderr[-3000:]}"
    return [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]


@pytest.mark.real_models
def test_words_arrive_while_speaking_and_the_final_text_is_whole(tmp_path):
    wav = _clip(tmp_path)
    # stdin is closed at once, which is "stop": the worker must still finish the audio it holds
    lines = _run(wav, {"LOCALCODE_CONTROL_URL": ""})
    assert lines[0] == {"type": "recording"} and lines[-1]["type"] == "final"

    proc = subprocess.Popen([str(VOICE_PY), str(WORKER), str(MODEL), "--wav", str(wav)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, env={**os.environ, "LOCALCODE_CONTROL_URL": ""})
    seen = [json.loads(line) for line in proc.stdout if line.startswith("{")]        # the clip ends by itself
    proc.wait(timeout=60)
    partials = [m["text"] for m in seen if m["type"] == "partial"]
    final = seen[-1]["text"].lower()
    # how many partials arrive depends on how busy the GPU is; one is enough to show the text came early
    assert partials, "text must arrive before the end"
    assert any("calculator" in p.lower() for p in partials)
    assert "calculator" in final and "test suite" in final


@pytest.mark.real_models
@pytest.mark.usefixtures("loopback_listener")
def test_nothing_is_transcribed_while_the_coding_model_is_busy(tmp_path):
    wav = _clip(tmp_path)

    class Busy(BaseHTTPRequestHandler):
        asked = 0

        def do_GET(self):
            Busy.asked += 1
            body = json.dumps({"phase": "generating"}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Busy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        proc = subprocess.Popen([str(VOICE_PY), str(WORKER), str(MODEL), "--wav", str(wav)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True,
                                env={**os.environ, "LOCALCODE_CONTROL_URL": f"http://127.0.0.1:{server.server_port}", "LOCALCODE_CONTROL_TOKEN": "t"})
        seen = [json.loads(line) for line in proc.stdout if line.startswith("{")]
        proc.wait(timeout=60)
    finally:
        server.shutdown()
    assert Busy.asked >= 2
    assert [m for m in seen if m["type"] == "partial"] == []          # the GPU was left to the coding model
    assert "calculator" in seen[-1]["text"].lower()                    # and nothing was lost: it is all in the final text


@pytest.mark.parametrize('gpu', [None, '0', '1'])
def test_worker_finalizes_on_cpu_or_default_backend(monkeypatch, capsys, gpu):
    """Exercise the worker protocol without a native speech model or GPU."""
    import contextlib
    import io
    import types
    np = pytest.importorskip('numpy')
    from localcode import voice, voice_live as worker

    if gpu is None:
        monkeypatch.delenv('LOCALCODE_VOICE_GPU', raising=False)
    else:
        monkeypatch.setenv('LOCALCODE_VOICE_GPU', gpu)
    monkeypatch.setenv('LOCALCODE_CONTROL_URL', '')
    monkeypatch.setattr(sys, 'argv', ['voice_live.py', 'model.bin', '--wav', 'unused.wav'])
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))
    monkeypatch.setattr(voice, '_silence_native_stderr', contextlib.nullcontext)
    loaded = {}

    class Model:
        def __init__(self, path, **options):
            loaded.update(options)

        def transcribe(self, samples):
            assert len(samples) == worker.RATE
            return [types.SimpleNamespace(text='Open the calculator')]

    class Audio:
        def __init__(self):
            self.done = threading.Event()
            self.done.set()

        def samples(self):
            t = np.arange(worker.RATE) / worker.RATE
            return (np.sin(2 * np.pi * 220 * t) * 6000).astype('int16')

    monkeypatch.setitem(sys.modules, 'pywhispercpp.model', types.SimpleNamespace(Model=Model))
    monkeypatch.setattr(worker, 'Audio', Audio)
    monkeypatch.setattr(worker, 'replay', lambda *args: None)
    assert worker.main() == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines == [{'type': 'recording'}, {'type': 'final', 'text': 'Open the calculator'}]
    assert loaded.get('context_params') == ({'use_gpu': False} if gpu == '0' else None)
