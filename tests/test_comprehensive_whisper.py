"""REAL whisper speech-to-text test (opt-in).

test_comprehensive_voice.py mocks pywhispercpp so the STT *logic* runs on
every push. This test instead runs whisper.cpp FOR REAL: it synthesises
speech with macOS `say`, feeds the audio through the actual transcription
path, and asserts the words come back. That needs the ~540 MB whisper
model and the `pywhispercpp` wheel, so it's opt-in:

    LOCALCODE_RUN_WHISPER_TEST=1 pytest tests/test_comprehensive_whisper.py -v

Skipped by default, and auto-skipped if the deps/binaries aren't present.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import voice
from localcode.voice import VoiceState

_GATED = os.environ.get("LOCALCODE_RUN_WHISPER_TEST") != "1"


def _pywhispercpp_available() -> bool:
    try:
        import pywhispercpp.model  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(_GATED, reason="set LOCALCODE_RUN_WHISPER_TEST=1 to run real whisper STT"),
    pytest.mark.skipif(sys.platform != "darwin", reason="uses macOS `say` to synthesise speech"),
    pytest.mark.skipif(shutil.which("say") is None, reason="`say` binary not found"),
    pytest.mark.skipif(not _pywhispercpp_available(),
                       reason="pywhispercpp not installed (pip install pywhispercpp)"),
]


def _synthesize_speech(text: str, out_wav: Path) -> None:
    """Render `text` to a 16 kHz mono WAV — whisper's native input rate."""
    subprocess.run(
        ["say", "-o", str(out_wav), "--data-format=LEI16@16000", text],
        check=True, timeout=30,
    )


@pytest.fixture(scope="module")
def ready_state():
    """A VoiceState with the whisper model downloaded (downloads on first run)."""
    state = VoiceState()
    if not voice.stt_model_ready(state):
        ok, msg = voice.ensure_stt_model(state)
        assert ok, f"failed to fetch whisper model: {msg}"
    return state


def test_real_transcription_roundtrip(ready_state, tmp_path):
    wav = tmp_path / "speech.wav"
    _synthesize_speech("testing one two three four five", wav)
    assert wav.exists() and wav.stat().st_size > 1000

    ok, text = voice.transcribe(ready_state, wav)
    assert ok, text
    lowered = text.lower()
    # Whisper renders spoken numbers EITHER as words ("one") or numerals
    # ("1") — e.g. "testing one two three" often comes back "Testing 123".
    # Accept both forms.
    word_hits = sum(w in lowered for w in ("one", "two", "three", "four", "five"))
    digit_hits = sum(d in lowered for d in ("1", "2", "3", "4", "5"))
    assert "testing" in lowered, f"missing 'testing': {text!r}"
    assert max(word_hits, digit_hits) >= 3, f"transcription too far off: {text!r}"


def test_real_transcription_of_a_sentence(ready_state, tmp_path):
    wav = tmp_path / "sentence.wav"
    _synthesize_speech("open the main python file", wav)
    ok, text = voice.transcribe(ready_state, wav)
    assert ok, text
    lowered = text.lower()
    assert "python" in lowered or "main" in lowered, f"got: {text!r}"
