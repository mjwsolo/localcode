from __future__ import annotations

import importlib.util
import shutil

from .config import AppConfig, RuntimeConfig


def provider_readiness(config: RuntimeConfig) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True
    provider = config.provider
    if provider == "ollama":
        if shutil.which("ollama") is None:
            ok = False
            messages.append("Ollama CLI missing. Install it or run `gem setup --install`.")
        if not config.model:
            messages.append("No explicit Ollama model tag set; Gem will use the profile default.")
        return ok, messages
    if provider == "mlx-local":
        if importlib.util.find_spec("mlx_lm") is None:
            ok = False
            messages.append("`mlx-lm` is not installed. Run `gem setup --install` or `pip install -U mlx-lm`.")
        if not (config.mlx_model_id or config.model):
            ok = False
            messages.append("No MLX model id configured. Set runtime.mlx_model_id.")
        return ok, messages
    if provider == "huggingface-local":
        missing = [name for name in ("transformers", "torch", "accelerate") if importlib.util.find_spec(name) is None]
        if missing:
            ok = False
            messages.append(f"Missing HF local packages: {', '.join(missing)}.")
        if not (config.huggingface_model_id or config.model):
            ok = False
            messages.append("No Hugging Face model id configured. Set runtime.huggingface_model_id.")
        return ok, messages
    if provider == "llama_cpp":
        if shutil.which("llama-server") is None and shutil.which("llama_cpp.server") is None:
            messages.append("No local llama.cpp server binary detected; run `gem setup --install` if supported, or install/start a local server manually.")
        if not config.base_url:
            ok = False
            messages.append("No llama.cpp server URL configured.")
        if not config.model:
            messages.append("No explicit llama.cpp model id set.")
        return ok, messages
    messages.append(f"Unknown provider: {provider}")
    return False, messages


def browser_voice_readiness(config: AppConfig) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True
    if config.browser.enabled:
        if shutil.which(config.browser.launch_command) is None:
            ok = False
            messages.append(
                f"Browser launcher `{config.browser.launch_command}` is missing. Install Node.js/npm so Playwright MCP can run."
            )
        messages.append(
            f"Browser MCP preset: {config.browser.mcp_server_name} -> {config.browser.launch_command} {' '.join(config.browser.launch_args or [])}"
        )
    if config.voice.stt_provider == "whisper.cpp":
        if shutil.which("whisper-cli") is None:
            ok = False
            messages.append("STT default `whisper.cpp` is missing. Install whisper.cpp and expose `whisper-cli` on PATH.")
        if not config.voice.whisper_model_path:
            messages.append("No whisper.cpp model configured. Set voice.whisper_model_path for offline transcription.")
    elif config.voice.stt_provider == "faster-whisper":
        if importlib.util.find_spec("faster_whisper") is None:
            ok = False
            messages.append("STT provider `faster-whisper` is missing. Install it with `pip install faster-whisper`.")
    if config.voice.tts_provider == "kokoro":
        if importlib.util.find_spec("kokoro") is None:
            ok = False
            messages.append("TTS default `kokoro` is missing. On Python 3.13, prefer `piper`; on 3.11/3.12, install with `pip install kokoro soundfile`.")
    elif config.voice.tts_provider == "piper":
        if shutil.which("piper") is None:
            ok = False
            messages.append("TTS provider `piper` is missing. Install Piper and expose `piper` on PATH.")
        if not config.voice.piper_model_path:
            messages.append("No Piper model configured. Set voice.piper_model_path.")
    return ok, messages
