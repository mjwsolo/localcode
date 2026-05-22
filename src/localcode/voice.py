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
    # Default engine is Piper — high-quality neural voice, free, fully
    # local, auto-downloads ~70 MB voice file on first use. macOS `say`
    # is kept as a fallback when Piper deps aren't installed.
    tts_engine: str = "piper"          # "piper" | "say" | "off"
    tts_voice: str | None = "en_US-amy-medium"  # default Piper voice id
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
import threading as _threading


# Single global lock around fd-2 redirection. Without it, two threads
# (e.g. streaming-tick worker + final-transcribe worker) can BOTH enter
# `_silence_native_stderr` concurrently — the second `dup(2)` would
# capture the FIRST thread's already-swapped /dev/null fd as its
# "original", and restoring it sends real stderr permanently to
# /dev/null + leaks file descriptors. That's been the source of the
# "second voice input crashes the TUI" failures.
_STDERR_REDIRECT_LOCK = _threading.Lock()

# Also serialize transcribe() calls — whisper.cpp Metal's cached
# Model isn't designed for concurrent transcribe calls (the kv cache
# inside the model is per-model, not per-call). Two concurrent calls
# can crash the Metal backend on some buffer shapes. Locking here is
# defense in depth on top of the fd-redirect lock.
_TRANSCRIBE_LOCK = _threading.Lock()


@contextlib.contextmanager
def _silence_native_stderr():
    """Redirect fd 2 to /dev/null around a block so C/C++ libraries
    (whisper.cpp, ggml, Metal, PortAudio) can't write to the terminal
    and corrupt textual's altscreen rendering.

    Thread-safe: only one redirection at a time. Concurrent callers
    block on the lock. Without the lock the dup2-pair races and we
    end up with a permanently-redirected fd 2 + leaked file
    descriptors — which manifests as voice-mode-second-press crashes.
    """
    with _STDERR_REDIRECT_LOCK:
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

# Whisper emits non-speech annotations in TWO syntaxes:
#   - [BLANK_AUDIO]  [NON-ENGLISH SPEECH]  [MUSIC]  [INAUDIBLE]  [SILENCE]
#   - *Singing*  *Music playing*  *Applause*  *whispers*  *laughter*
# Both are model-internal artifacts and should never reach the input box.
# We strip every bracketed [TAG] (caps + spaces + underscores) AND every
# starred *Tag* (any word starting with a letter, followed by letters/
# spaces, between asterisks).
_WHISPER_ANNOTATION_BRACKETS = _re.compile(r"\[[A-Z][A-Z0-9 _\-]*\]")
_WHISPER_ANNOTATION_STARS = _re.compile(r"\*[A-Za-z][A-Za-z _\-]*\*")


def _clean_transcript(text: str) -> str:
    """Remove Whisper's non-speech annotations + collapse whitespace."""
    text = _WHISPER_ANNOTATION_BRACKETS.sub("", text or "")
    text = _WHISPER_ANNOTATION_STARS.sub("", text)
    return " ".join(text.split())


def detect_voice_capability() -> tuple[bool, str]:
    """Return (ok, hint) — can this machine actually use voice mode?

    Checks, in order:
      1. macOS-only (Whisper.cpp via Metal is Mac-only in our build)
      2. At least one audio input device present (no mic = no voice)
      3. Terminal capability (Info.plist mic usage descriptor)
      4. Free disk for the ~514 MB Whisper download

    Returns (False, "human-readable reason") when voice cannot work.
    The chat handler shows the reason in the log before downloading.
    """
    import platform, shutil
    if platform.system() != "Darwin":
        return False, "Voice mode is currently macOS-only (Whisper.cpp Metal build)."
    # Mic check via sounddevice
    try:
        with _silence_native_stderr():
            import sounddevice as _sd
            devs = _sd.query_devices()
            inputs = [d for d in devs if d.get("max_input_channels", 0) > 0]
        if not inputs:
            return False, (
                "No audio input device found. Plug in a mic or set a default "
                "input in System Settings → Sound."
            )
    except Exception as e:
        return False, f"Audio backend (PortAudio) couldn't enumerate devices: {e}"
    # Disk
    try:
        free_bytes = shutil.disk_usage(str(DEFAULT_STT_MODEL_DIR.parent)).free
        if free_bytes < 700 * 1024 * 1024:  # 700 MB safety margin over 514 MB
            return False, (
                f"Less than 700 MB free disk — Whisper model needs ~514 MB. "
                f"Currently {free_bytes // (1024*1024)} MB free."
            )
    except Exception:
        pass
    # Terminal capability
    ok, hint = host_terminal_supports_mic()
    if not ok:
        return False, hint
    return True, ""


def detect_vision_capability() -> tuple[bool, str]:
    """Return (ok, hint) — can this machine load a vision projector?

    Checks free unified memory — mmproj sidecars are ~600 MB to 1.2 GB
    and load alongside the text decoder. If the machine is already
    tight on RAM, refusing here is friendlier than letting llama-server
    OOM-kill mid-load.
    """
    import subprocess, platform
    if platform.system() != "Darwin":
        return False, "Vision mode is currently Apple-Silicon-only (Metal mmproj)."
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        ram_gb = int(out) // (1024 ** 3)
        if ram_gb < 16:
            return False, (
                f"Vision needs ~16 GB unified memory; this Mac has {ram_gb} GB. "
                "Stick to text-only mode."
            )
    except Exception:
        pass
    return True, ""


def host_terminal_supports_mic() -> tuple[bool, str]:
    """Detect if the terminal hosting localcode can request mic access.

    macOS shows the native "Allow Mic?" dialog only if the parent app
    (the terminal) declares NSMicrophoneUsageDescription in its
    Info.plist. Terminal.app + iTerm + Ghostty + Alacritty have it;
    VS Code's integrated terminal historically does NOT — so PortAudio
    fails silently with "permission denied" and the user never sees a
    system prompt.

    Returns (ok, hint). When `ok=False`, hint explains the issue.
    """
    import platform
    if platform.system() != "Darwin":
        return True, ""
    # Walk up the process tree looking for a terminal app bundle.
    try:
        import subprocess as _sp
        pid = os.getppid()
        for _ in range(8):  # don't loop forever
            out = _sp.run(
                ["ps", "-o", "comm=,ppid=", "-p", str(pid)],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            if not out:
                break
            parts = out.rsplit(None, 1)
            if len(parts) != 2:
                break
            comm, ppid_str = parts[0], parts[1]
            lower = comm.lower()
            if "code helper" in lower or "code.app" in lower:
                return False, (
                    "Running inside VS Code's integrated terminal. VS Code's "
                    "Info.plist doesn't declare microphone usage, so macOS "
                    "won't show the permission dialog and PortAudio will fail "
                    "silently. Open Terminal.app or iTerm instead, run "
                    "`localcode`, and the system will ask for mic access on "
                    "first /voice."
                )
            if (
                "terminal.app" in lower or "iterm" in lower
                or "ghostty" in lower or "alacritty" in lower
                or "kitty" in lower or "warp" in lower
            ):
                return True, ""
            if ppid_str == "1" or ppid_str == "0":
                break
            try:
                pid = int(ppid_str)
            except ValueError:
                break
    except Exception:
        pass
    return True, ""  # unknown — let the OS prompt or PortAudio surface real error


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
    # Serialize transcribe calls. The cached Model holds per-instance
    # state (Metal KV cache, decoder buffers); two concurrent calls
    # can corrupt that state and crash the Metal backend. The lock
    # makes the cost a bit higher when streaming + final overlap but
    # eliminates the second-press crashes the user kept hitting.
    with _TRANSCRIBE_LOCK:
        try:
            with _silence_native_stderr():
                if _CACHED_MODEL is None or _CACHED_MODEL_PATH != model_path:
                    _CACHED_MODEL = Model(
                        str(model_path),
                        n_threads=max(2, (os.cpu_count() or 4) // 2),
                        print_realtime=False,
                        print_progress=False,
                        print_timestamps=False,
                    )
                    _CACHED_MODEL_PATH = model_path
                segments = _CACHED_MODEL.transcribe(str(audio_wav_path))
            text = " ".join(s.text for s in segments).strip()
            text = _clean_transcript(text)
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
        # Path A — sounddevice (in-process; cleanest). PortAudio prints
        # its CoreAudio errors to stderr fd=2 which corrupts textual's
        # altscreen; silence it the same way we silence whisper.cpp.
        self._sd_error: str | None = None
        try:
            with _silence_native_stderr():
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
        except Exception as e:
            self._sd_recording = None
            self._stream = None
            # Capture the real reason for the caller (chat handler)
            # so it can show a useful message instead of the generic
            # "no backend found".
            err = str(e).lower()
            if "permission" in err or "not permitted" in err or "denied" in err:
                self._sd_error = (
                    "macOS denied microphone access. Open System Settings → "
                    "Privacy & Security → Microphone and enable it for your "
                    "terminal (Terminal.app, iTerm, VS Code, etc.), then "
                    "restart localcode."
                )
            elif "device unavailable" in err or "invalid device" in err or "not found" in err:
                self._sd_error = (
                    "No input device found. Plug in a mic or check that your "
                    "system default input device is set in System Settings → Sound."
                )
            else:
                self._sd_error = f"sounddevice/PortAudio error: {e}"
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
        # If sounddevice was importable but threw on init, surface
        # THAT reason (mic permission / no device) instead of the
        # generic "install one of" pitch — the deps ARE installed.
        if self._sd_error:
            raise RuntimeError(self._sd_error)
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
        # Piper-first: try the natural neural voice. If anything goes
        # wrong (deps missing, voice file unreachable, audio I/O fails),
        # fall back to macOS `say` so the user still hears something
        # rather than silent failure.
        try:
            if engine == "piper":
                ok = _speak_piper(text, state)
                if ok:
                    return
            # macOS `say` fallback (also the explicit engine="say" path)
            if platform.system() == "Darwin":
                cmd = ["say"]
                # Only forward a voice arg if it looks like a `say`-style
                # voice name (Piper IDs contain ":" or "-" patterns that
                # would confuse say). Leave it unset for default Samantha.
                v = state.tts_voice or ""
                if v and "-" not in v.split("_", 1)[-1]:
                    cmd.extend(["-v", v])
                cmd.append(text)
                subprocess.run(cmd, check=False)
        except Exception:
            pass

    threading.Thread(target=_runner, daemon=True).start()


PIPER_VOICE_DIR = Path.home() / ".local" / "share" / "localcode" / "voice" / "piper"


def _ensure_piper_voice(voice_id: str) -> Path | None:
    """Download a Piper voice model (.onnx + .json) on first use.

    `voice_id` examples: "en_US-amy-medium", "en_US-libritts_r-medium",
    "en_GB-alan-medium". Full catalog at
    https://github.com/rhasspy/piper/blob/master/VOICES.md
    """
    PIPER_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    onnx = PIPER_VOICE_DIR / f"{voice_id}.onnx"
    cfg = PIPER_VOICE_DIR / f"{voice_id}.onnx.json"
    if onnx.is_file() and cfg.is_file():
        return onnx
    # Parse e.g. "en_US-amy-medium" → lang "en", country "US", voice "amy", quality "medium"
    try:
        lang_country, voice, quality = voice_id.split("-", 2)
        lang, country = lang_country.split("_", 1)
    except ValueError:
        return None
    base = (
        f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{lang}/{lang_country}/{voice}/{quality}"
    )
    import urllib.request
    try:
        if not onnx.is_file():
            urllib.request.urlretrieve(f"{base}/{voice_id}.onnx", str(onnx))
        if not cfg.is_file():
            urllib.request.urlretrieve(f"{base}/{voice_id}.onnx.json", str(cfg))
        return onnx
    except Exception:
        # Clean up partial files so next attempt is fresh.
        for p in (onnx, cfg):
            try:
                if p.is_file() and p.stat().st_size < 1024 * 1024:
                    p.unlink()
            except Exception:
                pass
        return None


def _speak_piper(text: str, state: VoiceState) -> bool:
    """Piper TTS — natural neural voice, free, fully local.

    Returns True if playback started, False on any failure (so the
    caller can fall back to macOS `say`). Auto-downloads the .onnx +
    .json voice file on first use into the piper voice cache.
    """
    try:
        from piper import PiperVoice  # type: ignore
    except ImportError:
        return False
    if not state.tts_voice:
        return False
    onnx_path = _ensure_piper_voice(state.tts_voice)
    if onnx_path is None:
        return False
    try:
        voice = PiperVoice.load(str(onnx_path))
        import sounddevice as sd
        import numpy as np
        sample_rate = voice.config.sample_rate
        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as out:
            for chunk in voice.synthesize_stream_raw(text):
                arr = np.frombuffer(chunk, dtype="int16")
                out.write(arr)
        return True
    except Exception:
        return False


def stop_speaking() -> None:
    """Best-effort interrupt of any in-flight TTS playback. macOS `say`
    can be killed via pkill; piper streams die when we close their thread."""
    if platform.system() == "Darwin":
        subprocess.run(["pkill", "-9", "say"], check=False, capture_output=True)
