"""Local voice I/O — push-to-talk speech-to-text + spoken responses.

Both halves are local-first / open-source / free:

- **STT** uses whisper.cpp via the `pywhispercpp` pip wheel, which ships
  prebuilt Metal binaries on Apple Silicon. Default model is
  `distil-medium.en` (~370 MB, ~15x realtime on M5 Max, English).

- **TTS** defaults to macOS `say` (ships with macOS, zero deps). Piper
  TTS is a documented opt-in upgrade via `runtime.tts_engine = "piper"`.

This module intentionally does NOT auto-record on import — it only
provides the building blocks. The TUI's chat screen owns the keybinding
(push-to-talk) and decides when to start/stop recording.

Permissions: first capture triggers macOS microphone TCC prompt. No
other system permissions required.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_STT_MODEL_DIR = Path.home() / ".local" / "share" / "localcode" / "voice"
# ggerganov/whisper.cpp hosts canonical ggml-format whisper weights. The
# `medium.en-q5_0` variant is the sweet spot for English coding/dictation:
# ~514 MB, ~15× realtime on M5 Max, accurate enough for tool/function names.
# An earlier version of this file referenced a `distil-whisper/distil-*`
# filename that 404'd — those weights only ship in safetensors/pytorch
# format, not GGUF. Fixed 2026-05-22.
DEFAULT_STT_REPO = "ggerganov/whisper.cpp"
DEFAULT_STT_MODEL_NAME = "ggml-medium.en-q5_0.bin"
DEFAULT_STT_MODEL_SIZE_MB = 514
DEFAULT_STT_MODEL_URL = (
    f"https://huggingface.co/{DEFAULT_STT_REPO}/resolve/main/{DEFAULT_STT_MODEL_NAME}"
)


@dataclass
class VoiceState:
    """Process-wide voice settings. Held on AppConfig in the actual app;
    this dataclass is the in-memory shape."""
    enabled: bool = False
    stt_model_path: Path | None = None
    tts_engine: str = "say"            # "say" | "piper" | "off"
    tts_voice: str | None = None       # macOS voice name (`say -v ?`) when engine="say"
    # Default "off" so audio doesn't auto-play just because the user
    # enabled voice mode — TTS is a separate /audio toggle now.
    tts_speak_mode: str = "off"        # "off" | "final" | "always"
    # Push-to-talk key. Default is Space: tap once to start recording,
    # the silence detector (1.5 s of quiet) auto-stops + transcribes —
    # so the user experience is "talk and stop", indistinguishable from
    # true hold-to-talk. Terminals can't actually report key release
    # (no SIGRELEASE in TTY), so true hold-to-release isn't physically
    # possible without OS-level keyboard hooks. We intercept Space at
    # the chat screen level when voice mode is ON; when OFF, Space
    # types normally into the input box.
    ptt_key: str = "space"
    sample_rate_hz: int = 16000        # whisper's native input rate
    # Audio capture device (None = system default). Honored by sounddevice
    # when present; ignored by the ffmpeg fallback path.
    input_device: str | None = None


# ─────────────────────────── STT ───────────────────────────

def stt_model_ready(state: VoiceState) -> bool:
    """Is the configured STT model on disk and large enough to be real?"""
    p = state.stt_model_path or (DEFAULT_STT_MODEL_DIR / DEFAULT_STT_MODEL_NAME)
    try:
        return p.is_file() and p.stat().st_size > 10 * 1024 * 1024
    except OSError:
        return False


def ensure_stt_model(state: VoiceState,
                     on_progress: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """Download the STT model if not already present. Returns (ok, path_or_err).

    Routes through the same `download_model` machinery as LLM downloads,
    so we automatically get:
      - hf_transfer fast path with urllib fallback
      - 3-attempt retry with exponential backoff (0/2/5 s)
      - partial-file preservation between attempts (resume)
      - error categorization (disk_full / auth / not_found / network / ssl)
      - clear user-facing remediation messages

    Implementation trick: build a lightweight in-memory ModelChoice that
    points at the whisper GGUF, then call download_model() on it. The
    catalog never sees this entry — it's a private proxy used purely to
    reuse the download pipeline.
    """
    target_path = state.stt_model_path or (DEFAULT_STT_MODEL_DIR / DEFAULT_STT_MODEL_NAME)
    if target_path.is_file() and target_path.stat().st_size > 10 * 1024 * 1024:
        return True, str(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Proxy ModelChoice pointing at the whisper file in the voice dir.
    from .models_catalog import ModelChoice
    proxy = ModelChoice(
        key="_voice_stt",
        name="Whisper medium.en (q5_0)",
        hf_repo=DEFAULT_STT_REPO,
        filename=DEFAULT_STT_MODEL_NAME,
        size_gb=DEFAULT_STT_MODEL_SIZE_MB / 1024.0,
        active_params="—",
        architecture="whisper",
        license="MIT (OpenAI Whisper weights)",
        humaneval_pass_at_1=None,
        notes="Voice STT — one-time download for /voice.",
    )

    # download_model expects the target file under model_dir(), but we
    # want whisper in DEFAULT_STT_MODEL_DIR. Trick: temporarily monkey-
    # patch model_dir for this one call. Cleaner than threading a custom
    # destination through the whole pipeline.
    from . import models_catalog as _mc
    orig_md = _mc.model_dir
    try:
        _mc.model_dir = lambda: target_path.parent
        from .bootstrap import download_model
        ok, result = download_model(proxy, on_progress=on_progress)
    finally:
        _mc.model_dir = orig_md
    return ok, result


import contextlib
import sys


@contextlib.contextmanager
def _silence_native_stderr():
    """Redirect file descriptor 2 to /dev/null around a block so the
    C/C++ libraries underneath us (whisper.cpp, ggml, Metal) can't
    write directly to the terminal and corrupt textual's altscreen
    rendering. Python's `sys.stderr` redirection won't catch native
    writes — only an fd-level dup does.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            os.dup2(saved, 2)
        finally:
            os.close(devnull)
            os.close(saved)


# Process-wide cached whisper.cpp Model. Loading the model is ~538 MB +
# ~1 s of init time per call; once loaded it can transcribe many WAVs
# without re-allocating. Reset to None if the state.stt_model_path
# changes.
_CACHED_MODEL: object | None = None
_CACHED_MODEL_PATH: Path | None = None


import re as _re

# Whisper emits bracketed annotations for non-speech audio
# ("[BLANK_AUDIO]", "[NON-ENGLISH SPEECH]", "[MUSIC]", "[INAUDIBLE]",
# etc.). These are model-internal artifacts that should never end up
# in the user's input box. We strip any bracketed all-caps tag (with
# spaces, underscores, hyphens allowed) — that catches every known
# whisper annotation while leaving normal user text alone.
_WHISPER_ANNOTATION_RE = _re.compile(r"\[[A-Z][A-Z0-9 _\-]*\]")


def _clean_transcript(text: str) -> str:
    """Remove Whisper's non-speech annotations + collapse whitespace."""
    text = _WHISPER_ANNOTATION_RE.sub("", text or "")
    return " ".join(text.split())


def transcribe(state: VoiceState, audio_wav_path: Path) -> tuple[bool, str]:
    """Transcribe a WAV file with whisper.cpp via pywhispercpp.

    Returns (ok, text_or_error). Falls back to a clear error if
    pywhispercpp isn't installed — the chat-layer caller surfaces it
    to the user with the install hint.

    The model is loaded ONCE per process (cached in module state) so
    streaming-style repeat calls every 1.5 s don't keep re-allocating
    540 MB. All native stderr from whisper/ggml/Metal is suppressed so
    it can't corrupt textual's altscreen rendering.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    if not stt_model_ready(state):
        return False, "STT model not downloaded. Run /voice setup first."
    try:
        from pywhispercpp.model import Model
    except ImportError:
        return False, (
            "pywhispercpp not installed. Add 'pywhispercpp' to deps or "
            "run: pip install pywhispercpp"
        )
    model_path = state.stt_model_path or (DEFAULT_STT_MODEL_DIR / DEFAULT_STT_MODEL_NAME)
    try:
        with _silence_native_stderr():
            if _CACHED_MODEL is None or _CACHED_MODEL_PATH != model_path:
                _CACHED_MODEL = Model(
                    str(model_path),
                    n_threads=max(2, (os.cpu_count() or 4) // 2),
                    # pywhispercpp also accepts these decoder flags to
                    # silence its per-segment progress prints — bools
                    # default True in some builds.
                    print_realtime=False,
                    print_progress=False,
                    print_timestamps=False,
                )
                _CACHED_MODEL_PATH = model_path
            segments = _CACHED_MODEL.transcribe(str(audio_wav_path))
        text = " ".join(s.text for s in segments).strip()
        text = _clean_transcript(text)  # strip [NON-ENGLISH SPEECH] etc.
        return True, text
    except Exception as e:
        return False, f"Transcription failed: {e}"


# ─────────────────────────── Audio capture ───────────────────────────

class Recorder:
    """Background audio recorder. Start → stop → returns a wav file path.

    Uses sounddevice if available (cleanest). Falls back to ffmpeg / sox
    via subprocess if not. We DO NOT pull pyaudio — it's a maintenance
    nightmare on macOS Python builds.

    Live-level fields (`peak`, `rms`, `silence_seconds`) are updated in
    the sounddevice callback so a UI widget can poll them every frame
    without touching the audio buffer. The `peak` field is the maximum
    absolute sample value normalized to [0.0, 1.0] over the last
    callback chunk; `rms` is the root-mean-square; `silence_seconds`
    is the wall-clock time since the last sample exceeded
    `silence_threshold`. Used by the chat screen's visualizer + the
    optional auto-stop-on-silence handler.
    """

    # If the running RMS stays below this for `silence_window_s` seconds,
    # `silence_seconds` will surpass that window and callers can choose
    # to stop the recording.
    silence_threshold: float = 0.012

    # Hard cap on recording length so a forgotten Space-hold can't
    # grow the in-memory PCM buffer indefinitely. At 16 kHz mono int16
    # this is ~9.6 MB per minute; 5 min = ~48 MB which we'll reclaim
    # on the next stop().
    MAX_RECORDING_SECONDS: float = 300.0

    def __init__(self, state: VoiceState):
        self.state = state
        self._tmp_path: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._sd_recording = None  # sounddevice array if using that path
        # Live-level telemetry — read by the TUI visualizer
        self.peak: float = 0.0
        self.rms: float = 0.0
        import time as _t
        self._start_ts: float = _t.time()
        self._last_loud_ts: float = self._start_ts

    @property
    def silence_seconds(self) -> float:
        import time as _t
        # Don't fire silence detection in the first 400 ms — mic
        # initialization frequently dumps zero samples before audio
        # actually starts flowing. Without this grace window, recordings
        # auto-stopped before the user could say a single word.
        import time as _t
        elapsed = _t.time() - self._start_ts
        if elapsed < 0.4:
            return 0.0
        return _t.time() - self._last_loud_ts

    def start(self) -> None:
        if self._tmp_path is not None:
            return  # already recording — idempotent
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="localcode-ptt-")
        os.close(fd)
        self._tmp_path = Path(tmp)
        # Path A — sounddevice (in-process; cleanest)
        try:
            import sounddevice as sd
            import numpy as np  # noqa: F401
            self._sd_recording = []
            self._stream = sd.InputStream(
                samplerate=self.state.sample_rate_hz,
                channels=1,
                dtype="int16",
                callback=self._sd_callback,
            )
            self._stream.start()
            return
        except Exception:
            self._sd_recording = None
            self._stream = None
        # Path B — ffmpeg subprocess
        if shutil.which("ffmpeg"):
            self._proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "avfoundation", "-i", ":0",  # default mic
                    "-ar", str(self.state.sample_rate_hz),
                    "-ac", "1",
                    str(self._tmp_path),
                ],
                stdin=subprocess.PIPE,
            )
            return
        # Path C — sox
        if shutil.which("rec"):
            self._proc = subprocess.Popen(
                ["rec", "-q", "-r", str(self.state.sample_rate_hz),
                 "-c", "1", "-b", "16", str(self._tmp_path)],
            )
            return
        raise RuntimeError(
            "No audio capture backend found. Install one of: "
            "`pip install sounddevice numpy`, or `brew install ffmpeg`, "
            "or `brew install sox`."
        )

    def _sd_callback(self, indata, frames, time, status):
        if status:
            return  # surface later if needed
        if self._sd_recording is not None:
            self._sd_recording.append(indata.copy())
            # Hard cap so a forgotten hold can't OOM the process. Drop
            # the OLDEST chunks once we exceed MAX_RECORDING_SECONDS.
            max_chunks = int(
                self.MAX_RECORDING_SECONDS
                * self.state.sample_rate_hz
                / max(1, frames)
            )
            if len(self._sd_recording) > max_chunks:
                # Trim from the front (ring-buffer behavior).
                self._sd_recording = self._sd_recording[-max_chunks:]
        # Update live levels for the visualizer + silence detector.
        # indata is int16; convert to abs floats in [0,1].
        try:
            import numpy as np
            arr = indata.reshape(-1).astype("float32") / 32768.0
            self.peak = float(np.max(np.abs(arr)))
            self.rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
            if self.rms > self.silence_threshold:
                import time as _t
                self._last_loud_ts = _t.time()
        except Exception:
            pass

    def snapshot_wav(self) -> Path | None:
        """Write everything captured so far to a fresh WAV file WITHOUT
        stopping the stream. Returns the path, or None if there's nothing
        captured yet. Used by live-streaming transcription — caller can
        feed each snapshot to whisper and update the input field with
        the latest transcript while the user is still talking.
        """
        if self._sd_recording is None or not self._sd_recording:
            return None
        try:
            import numpy as np, wave, tempfile, os as _os
            buf = np.concatenate(self._sd_recording)
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="lc-stream-")
            _os.close(fd)
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.state.sample_rate_hz)
                wf.writeframes(buf.tobytes())
            return Path(tmp)
        except Exception:
            return None

    def stop(self) -> Path | None:
        """Stop recording, return the wav path."""
        if self._tmp_path is None:
            return None
        # sounddevice path
        if self._sd_recording is not None and getattr(self, "_stream", None):
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            try:
                import numpy as np
                import wave
                buf = np.concatenate(self._sd_recording) if self._sd_recording else np.zeros((0, 1), dtype="int16")
                with wave.open(str(self._tmp_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.state.sample_rate_hz)
                    wf.writeframes(buf.tobytes())
            except Exception:
                pass
            self._sd_recording = None
            self._stream = None
        # subprocess path
        elif self._proc is not None:
            try:
                # ffmpeg listens for 'q' on stdin to exit cleanly
                if self._proc.stdin:
                    try:
                        self._proc.stdin.write(b"q\n")
                        self._proc.stdin.flush()
                    except Exception:
                        pass
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        result = self._tmp_path
        self._tmp_path = None
        return result


# ─────────────────────────── TTS ───────────────────────────

def speak(text: str, state: VoiceState) -> None:
    """Speak `text` according to the current TTS engine config.

    Non-blocking — spawns the speaker process in a background thread so
    the agent doesn't pause while audio plays. Safe to call concurrently;
    each call queues its own subprocess.
    """
    if not text or state.tts_engine == "off":
        return
    engine = state.tts_engine

    def _runner():
        try:
            if engine == "say" and platform.system() == "Darwin":
                cmd = ["say"]
                if state.tts_voice:
                    cmd.extend(["-v", state.tts_voice])
                cmd.append(text)
                subprocess.run(cmd, check=False)
            elif engine == "piper":
                _speak_piper(text, state)
            # Unknown engines silently no-op (treat like "off") — this is
            # voice OUTPUT and shouldn't crash the conversation.
        except Exception:
            pass

    threading.Thread(target=_runner, daemon=True).start()


def _speak_piper(text: str, state: VoiceState) -> None:
    """Optional upgrade path. Requires `pip install piper-tts` and a
    downloaded voice model. See https://github.com/rhasspy/piper."""
    try:
        from piper import PiperVoice  # type: ignore
    except ImportError:
        return  # silently fall back to no-op; user can switch via /voice
    # Voice file is whatever the user configured via tts_voice (full path
    # to a .onnx voice model + sibling .json). Piper does its own audio
    # output via sounddevice or aplay.
    if not state.tts_voice:
        return
    try:
        voice = PiperVoice.load(state.tts_voice)
        # Stream audio to default output
        import sounddevice as sd
        sample_rate = voice.config.sample_rate
        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as out:
            for chunk in voice.synthesize_stream_raw(text):
                import numpy as np
                arr = np.frombuffer(chunk, dtype="int16")
                out.write(arr)
    except Exception:
        pass


def stop_speaking() -> None:
    """Best-effort interrupt of any in-flight TTS playback. macOS `say`
    can be killed via pkill; piper streams die when we close their thread."""
    if platform.system() == "Darwin":
        subprocess.run(["pkill", "-9", "say"], check=False, capture_output=True)
