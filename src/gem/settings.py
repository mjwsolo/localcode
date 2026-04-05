from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import AppConfig, save_config


VALID_KEYS = {
    "runtime.provider",
    "runtime.base_url",
    "runtime.profile",
    "runtime.model",
    "runtime.mode",
    "runtime.planner_model",
    "runtime.draft_model",
    "runtime.planner_enabled",
    "runtime.adaptive_execution",
    "runtime.escalation_enabled",
    "runtime.huggingface_model_id",
    "runtime.huggingface_device",
    "runtime.huggingface_dtype",
    "runtime.mlx_model_id",
    "runtime.quant_preset",
    "runtime.cache_policy",
    "runtime.rolling_window_messages",
    "runtime.llama_cpp_gpu_layers",
    "runtime.llama_cpp_threads",
    "runtime.llama_cpp_batch_size",
    "runtime.temperature",
    "runtime.max_context_chars",
    "runtime.request_timeout_seconds",
    "runtime.max_retries",
    "search.provider",
    "search.google_api_key",
    "search.google_cx",
    "search.brave_api_key",
    "search.serpapi_api_key",
    "browser.enabled",
    "browser.mcp_server_name",
    "browser.launch_command",
    "voice.stt_provider",
    "voice.tts_provider",
    "voice.whisper_model_path",
    "voice.faster_whisper_model",
    "voice.kokoro_voice",
    "voice.piper_model_path",
    "ui.show_debug",
    "ui.thinking_mode",
}


def show_settings(config: AppConfig) -> None:
    console = Console()
    table = Table("key", "value")
    for section_name in ("runtime", "search", "browser", "voice", "ui"):
        section = getattr(config, section_name)
        for key, value in asdict(section).items():
            table.add_row(f"{section_name}.{key}", str(value))
    console.print(table)


def set_setting(config: AppConfig, key: str, value: str) -> str:
    if key not in VALID_KEYS:
        raise ValueError(f"Unknown setting: {key}")
    section_name, field_name = key.split(".", 1)
    section = getattr(config, section_name)
    current = getattr(section, field_name)
    if isinstance(current, bool):
        parsed: Any = value.lower() in {"1", "true", "yes", "on"}
    elif isinstance(current, int):
        parsed = int(value)
    elif isinstance(current, float):
        parsed = float(value)
    else:
        parsed = value
    setattr(section, field_name, parsed)
    save_config(config)
    return f"Updated {key} to {parsed}"
