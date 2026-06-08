"""Audio I/O coverage — STT (speech→text) and TTS (text→speech).

Real voice needs a microphone, the whisper.cpp model (~540 MB), and the
`say` binary. None of that belongs in a unit run, so we mock the heavy
edges: `pywhispercpp` for transcription and `subprocess.Popen` for
playback. What's left — the actual transcript cleanup and TTS text
sanitisation logic, plus the engine dispatch — is real code.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import voice
from localcode.voice import VoiceState


# ── STT: transcript cleanup (pure logic) ────────────────────────────


def test_clean_transcript_applies_dictation_directives():
    cleaned = voice._clean_transcript(
        "hello comma world full stop capital z"
    )
    assert "," in cleaned          # "comma" → ","
    assert "." in cleaned          # "full stop" → "."
    assert "Z" in cleaned          # "capital z" → "Z"
    assert "comma" not in cleaned  # directive words consumed
    assert "full stop" not in cleaned


def test_clean_transcript_strips_whisper_annotations():
    assert voice._clean_transcript("[BLANK_AUDIO] hello (music) *laughs*").strip() == "hello"


# ── STT: transcribe() dependency edges ──────────────────────────────


def test_transcribe_errors_clearly_when_model_missing(tmp_path):
    state = VoiceState(stt_model_path=tmp_path / "nope.bin")
    ok, msg = voice.transcribe(state, tmp_path / "audio.wav")
    assert ok is False
    assert "not downloaded" in msg.lower()


def test_transcribe_success_with_mocked_whisper(tmp_path, monkeypatch):
    # A plausible-looking model file (>10 MB so stt_model_ready passes).
    model = tmp_path / "ggml.bin"
    with open(model, "wb") as f:
        f.seek(11 * 1024 * 1024)
        f.write(b"\0")
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF....WAVE")

    # Fake pywhispercpp.model.Model that "transcribes" to fixed segments.
    class _Seg:
        def __init__(self, text): self.text = text

    class _FakeModel:
        def __init__(self, *a, **k): pass
        def transcribe(self, _path): return [_Seg("hello"), _Seg("world")]

    fake_pkg = types.ModuleType("pywhispercpp")
    fake_mod = types.ModuleType("pywhispercpp.model")
    fake_mod.Model = _FakeModel
    fake_pkg.model = fake_mod
    monkeypatch.setitem(sys.modules, "pywhispercpp", fake_pkg)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", fake_mod)
    # Reset the module-level model cache so our fake is used.
    monkeypatch.setattr(voice, "_CACHED_MODEL", None, raising=False)
    monkeypatch.setattr(voice, "_CACHED_MODEL_PATH", None, raising=False)

    state = VoiceState(stt_model_path=model)
    ok, text = voice.transcribe(state, wav)
    assert ok is True
    assert text == "hello world"


# ── TTS: text sanitisation (pure logic) ─────────────────────────────


def test_strip_for_tts_replaces_code_and_links():
    out = voice._strip_for_tts(
        "Run ```python\nprint('hi')\n``` then see `x` at https://example.com/foo"
    )
    assert "(code)" in out
    assert "(link)" in out
    assert "print" not in out  # fenced code body removed
    assert "example.com" not in out


def test_strip_for_tts_drops_markdown_markers():
    out = voice._strip_for_tts("# Heading\n- **bold** point")
    assert "#" not in out
    assert "*" not in out
    assert "bold" in out and "point" in out


# ── TTS: speak() engine dispatch (mock the player) ──────────────────


def test_speak_off_is_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(voice.subprocess, "Popen", lambda *a, **k: calls.append(a))
    voice.speak("hello", VoiceState(tts_engine="off"))
    time.sleep(0.2)
    assert calls == []


def test_speak_say_spawns_subprocess_with_clean_text(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # keep tts.log inside sandbox
    monkeypatch.setattr(voice.platform, "system", lambda: "Darwin")

    spawned = []

    class _FakePopen:
        def __init__(self, cmd, *a, **k):
            spawned.append(cmd)
        def wait(self): return 0

    monkeypatch.setattr(voice.subprocess, "Popen", _FakePopen)

    voice.speak("Here is `code` to run.", VoiceState(tts_engine="say"))

    # speak() runs in a daemon thread — poll briefly for the spawn.
    deadline = time.time() + 3.0
    while not spawned and time.time() < deadline:
        time.sleep(0.02)

    # speak() first calls stop_speaking() (which may spawn `pkill`), then
    # the actual `say`. Find the say invocation among whatever was spawned.
    say_cmds = [c for c in spawned if c and c[0] == "say"]
    assert say_cmds, f"no `say` spawned (got {spawned})"
    spoken = say_cmds[0][-1]
    assert "(code)" in spoken  # inline code sanitised before speaking
    assert "`" not in spoken
