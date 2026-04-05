from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .config import AppConfig, ensure_home_dirs


def voice_status(config: AppConfig) -> list[str]:
    messages = [
        f"stt_provider={config.voice.stt_provider}",
        f"tts_provider={config.voice.tts_provider}",
    ]
    if config.voice.stt_provider == "whisper.cpp":
        messages.append("whisper-cli=present" if shutil.which("whisper-cli") else "whisper-cli=missing")
        messages.append(f"whisper_model_path={config.voice.whisper_model_path or '(unset)'}")
    elif config.voice.stt_provider == "faster-whisper":
        present = importlib.util.find_spec("faster_whisper") is not None
        messages.append("faster_whisper=present" if present else "faster_whisper=missing")
        messages.append(f"faster_whisper_model={config.voice.faster_whisper_model}")

    if config.voice.tts_provider == "kokoro":
        present = importlib.util.find_spec("kokoro") is not None
        messages.append("kokoro=present" if present else "kokoro=missing")
        messages.append(f"kokoro_voice={config.voice.kokoro_voice}")
    elif config.voice.tts_provider == "piper":
        messages.append("piper=present" if shutil.which("piper") else "piper=missing")
        messages.append(f"piper_model_path={config.voice.piper_model_path or '(unset)'}")
    return messages


def transcribe_audio(config: AppConfig, audio_path: str) -> str:
    path = Path(audio_path).expanduser()
    if not path.exists():
        return f"Audio file not found: {path}"
    if config.voice.stt_provider == "whisper.cpp":
        if shutil.which("whisper-cli") is None:
            return "whisper.cpp is not installed. Install whisper.cpp and ensure `whisper-cli` is on PATH."
        if not config.voice.whisper_model_path:
            return "No whisper.cpp model configured. Set voice.whisper_model_path."
        cmd = [
            "whisper-cli",
            "-m",
            config.voice.whisper_model_path,
            "-f",
            str(path),
            "-otxt",
            "-of",
            str(path.with_suffix("")),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        txt_path = path.with_suffix(".txt")
        transcript = txt_path.read_text(errors="replace") if txt_path.exists() else (result.stdout + "\n" + result.stderr).strip()
        return transcript or "No transcript returned."
    if config.voice.stt_provider == "faster-whisper":
        if importlib.util.find_spec("faster_whisper") is None:
            return "faster-whisper is not installed. Run `pip install faster-whisper`."
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel(config.voice.faster_whisper_model, device="auto", compute_type="int8")
        segments, _info = model.transcribe(str(path))
        return "\n".join(segment.text.strip() for segment in segments if segment.text.strip()) or "No transcript returned."
    return f"Unsupported STT provider: {config.voice.stt_provider}"


def speak_text(config: AppConfig, text: str) -> str:
    home = ensure_home_dirs()
    output_path = home / "audio" / "gem_voice.wav"
    if config.voice.tts_provider == "kokoro":
        if importlib.util.find_spec("kokoro") is None:
            return "Kokoro is not installed. Run `pip install kokoro soundfile`."
        script = (
            "from kokoro import KPipeline; import soundfile as sf; import sys; "
            "voice=sys.argv[1]; out=sys.argv[2]; text=sys.argv[3]; "
            "pipeline=KPipeline(lang_code='a'); gen=pipeline(text, voice=voice); "
            "audio=None\n"
            "for _, _, samples in gen:\n"
            "    audio=samples if audio is None else audio\n"
            "    break\n"
            "sf.write(out, audio, 24000); print(out)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, config.voice.kokoro_voice, str(output_path), text],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return (result.stdout + "\n" + result.stderr).strip() or "Kokoro synthesis failed."
        return f"Audio written to {output_path}"
    if config.voice.tts_provider == "piper":
        if shutil.which("piper") is None:
            return "Piper is not installed. Run `pip install piper-tts`."
        if not config.voice.piper_model_path:
            return "No Piper model configured. Set voice.piper_model_path."
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write(text)
            temp_input = handle.name
        result = subprocess.run(
            ["piper", "--model", config.voice.piper_model_path, "--output_file", str(output_path), "--input-file", temp_input],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return (result.stdout + "\n" + result.stderr).strip() or "Piper synthesis failed."
        return f"Audio written to {output_path}"
    return f"Unsupported TTS provider: {config.voice.tts_provider}"
