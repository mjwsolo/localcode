"""Gem audio input — load, resample, and format audio for Gemma 4 model input.

Gemma 4 audio specs (from official docs):
  - 16 kHz sample rate
  - Single channel (mono)
  - 32-bit float, samples in [-1, 1]
  - 25 tokens per second of audio
  - Maximum 30 seconds per clip

Supported message format (HuggingFace transformers):
  {"type": "audio", "audio": <path_or_numpy_array>}

For Ollama, audio is not yet natively supported in the API,
so we fall back to local transcription then send as text.
"""
from __future__ import annotations

import io
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 30
TOKENS_PER_SECOND = 25


@dataclass(slots=True)
class AudioData:
    """Audio ready to send to the model."""
    samples: list[float]  # float32 samples in [-1, 1]
    sample_rate: int
    duration_seconds: float
    source: str
    estimated_tokens: int

    @property
    def duration_str(self) -> str:
        return f"{self.duration_seconds:.1f}s"


def is_audio_file(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def load_audio(path: str) -> AudioData | None:
    """Load an audio file, resample to 16kHz mono float32.

    Uses ffmpeg for decoding (widely available), falls back to
    wave module for WAV files.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return None
    if p.suffix.lower() not in AUDIO_EXTENSIONS:
        return None

    samples = _load_with_ffmpeg(str(p))
    if samples is None:
        samples = _load_wav_stdlib(str(p))
    if samples is None:
        return None

    # Enforce max duration
    max_samples = TARGET_SAMPLE_RATE * MAX_DURATION_SECONDS
    if len(samples) > max_samples:
        samples = samples[:max_samples]

    duration = len(samples) / TARGET_SAMPLE_RATE
    return AudioData(
        samples=samples,
        sample_rate=TARGET_SAMPLE_RATE,
        duration_seconds=duration,
        source=str(p),
        estimated_tokens=int(duration * TOKENS_PER_SECOND),
    )


def _load_with_ffmpeg(path: str) -> list[float] | None:
    """Decode any audio format to 16kHz mono float32 using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", path,
                "-ar", str(TARGET_SAMPLE_RATE),  # resample to 16kHz
                "-ac", "1",                       # mono
                "-f", "f32le",                    # 32-bit float little-endian
                "-acodec", "pcm_f32le",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        raw = result.stdout
        if len(raw) < 4:
            return None
        # Unpack float32 samples
        num_samples = len(raw) // 4
        samples = list(struct.unpack(f"<{num_samples}f", raw))
        # Clamp to [-1, 1]
        samples = [max(-1.0, min(1.0, s)) for s in samples]
        return samples
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _load_wav_stdlib(path: str) -> list[float] | None:
    """Load a WAV file using Python's wave module (no external deps)."""
    import wave
    try:
        with wave.open(path, "rb") as wf:
            if wf.getnchannels() > 2:
                return None
            sample_width = wf.getsampwidth()
            if sample_width not in (1, 2, 4):
                return None
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            n_channels = wf.getnchannels()
            orig_rate = wf.getframerate()

            # Convert to float32
            if sample_width == 2:
                num_samples = len(raw) // 2
                int_samples = struct.unpack(f"<{num_samples}h", raw)
                float_samples = [s / 32768.0 for s in int_samples]
            elif sample_width == 4:
                num_samples = len(raw) // 4
                float_samples = list(struct.unpack(f"<{num_samples}f", raw))
            else:
                float_samples = [((b - 128) / 128.0) for b in raw]

            # Downmix stereo to mono
            if n_channels == 2:
                mono = []
                for i in range(0, len(float_samples), 2):
                    if i + 1 < len(float_samples):
                        mono.append((float_samples[i] + float_samples[i + 1]) / 2.0)
                    else:
                        mono.append(float_samples[i])
                float_samples = mono

            # Resample to 16kHz if needed
            if orig_rate != TARGET_SAMPLE_RATE:
                float_samples = _resample(float_samples, orig_rate, TARGET_SAMPLE_RATE)

            return float_samples
    except Exception:
        return None


def _resample(samples: list[float], from_rate: int, to_rate: int) -> list[float]:
    """Simple linear interpolation resampling."""
    if from_rate == to_rate:
        return samples
    ratio = from_rate / to_rate
    new_length = int(len(samples) / ratio)
    result = []
    for i in range(new_length):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx
        if idx + 1 < len(samples):
            val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        elif idx < len(samples):
            val = samples[idx]
        else:
            break
        result.append(max(-1.0, min(1.0, val)))
    return result


# ── Message formatting ───────────────────────────────────────────────────

def build_audio_message_hf(text: str, audio_path: str) -> dict:
    """Build a HuggingFace-compatible message with audio input.

    Format per Gemma 4 docs:
    {"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "audio", "audio": "path_to_file"}
    ]}
    """
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "audio", "audio": audio_path})
    return {"role": "user", "content": content}


def build_audio_message_with_samples(text: str, audio: AudioData) -> dict:
    """Build a message with pre-processed audio samples.

    Some processors accept numpy arrays directly.
    """
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    # Try to pass as numpy array if available, else pass the file path
    try:
        import numpy as np
        audio_array = np.array(audio.samples, dtype=np.float32)
        content.append({"type": "audio", "audio": audio_array})
    except ImportError:
        content.append({"type": "audio", "audio": audio.source})
    return {"role": "user", "content": content}


def detect_audio_paths_in_text(text: str) -> list[str]:
    """Find file paths in text that look like audio files."""
    import re
    pattern = r'(?:^|\s)([~/.]?[\w./-]+\.(?:' + "|".join(
        ext.lstrip(".") for ext in AUDIO_EXTENSIONS
    ) + r'))'
    return re.findall(pattern, text, re.IGNORECASE)


def audio_to_text_fallback(audio: AudioData, config: Any = None) -> str:
    """For providers that don't support native audio (Ollama), transcribe first.

    Uses whisper.cpp or faster-whisper if available.
    """
    # Try faster-whisper
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", compute_type="int8")
        import numpy as np
        audio_array = np.array(audio.samples, dtype=np.float32)
        segments, _ = model.transcribe(audio_array)
        return " ".join(seg.text for seg in segments).strip()
    except ImportError:
        pass

    # Fallback: write to temp WAV and use whisper CLI
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        _write_wav(tmp.name, audio.samples, TARGET_SAMPLE_RATE)
        result = subprocess.run(
            ["whisper", tmp.name, "--model", "small", "--output_format", "txt"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return f"[Audio: {audio.duration_str} from {audio.source} — transcription not available]"


def _write_wav(path: str, samples: list[float], sample_rate: int) -> None:
    """Write float32 samples to a WAV file."""
    import wave
    int_samples = [int(max(-1.0, min(1.0, s)) * 32767) for s in samples]
    raw = struct.pack(f"<{len(int_samples)}h", *int_samples)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
