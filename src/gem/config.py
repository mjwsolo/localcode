from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib


def default_tts_provider() -> str:
    if sys.version_info >= (3, 13):
        return "piper"
    return "kokoro"


DEFAULT_CONFIG = """[runtime]
provider = "ollama"
base_url = "http://localhost:11434"
profile = "e4b"
model = ""
mode = "balanced"
planner_model = "gemma4:e2b"
draft_model = "gemma4:e2b"
planner_enabled = true
adaptive_execution = true
escalation_enabled = true
huggingface_model_id = ""
huggingface_device = "auto"
huggingface_dtype = "auto"
mlx_model_id = ""
quant_preset = "balanced"
cache_policy = "adaptive"
rolling_window_messages = 24
llama_cpp_gpu_layers = 0
llama_cpp_threads = 8
llama_cpp_batch_size = 128
llama_cpp_spec_type = ""
llama_cpp_draft_max = 64
llama_cpp_expert_offload = false
llama_cpp_draft_model = ""
llama_cpp_lookup_cache = false
kv_cache_type = "q8_0"
temperature = 0.2
max_context_chars = 40000
request_timeout_seconds = 120
max_retries = 2

[search]
provider = "duckduckgo"
google_api_key = ""
google_cx = ""
brave_api_key = ""
serpapi_api_key = ""

[browser]
enabled = true
mcp_server_name = "browser"
launch_command = "npx"
launch_args = ["-y", "@playwright/mcp@latest"]

[voice]
stt_provider = "whisper.cpp"
tts_provider = "piper"
whisper_model_path = ""
faster_whisper_model = "small"
kokoro_voice = "af_heart"
piper_model_path = ""

[ui]
show_debug = false
thinking_mode = "full"
"""


@dataclass(slots=True)
class RuntimeConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    profile: str = "e4b"
    model: str = ""
    mode: str = "balanced"
    planner_model: str = "gemma4:e2b"
    draft_model: str = "gemma4:e2b"
    planner_enabled: bool = True
    adaptive_execution: bool = True
    escalation_enabled: bool = True
    huggingface_model_id: str = ""
    huggingface_device: str = "auto"
    huggingface_dtype: str = "auto"
    mlx_model_id: str = ""
    quant_preset: str = "balanced"
    cache_policy: str = "adaptive"
    rolling_window_messages: int = 24
    llama_cpp_gpu_layers: int = 0
    llama_cpp_threads: int = 8
    llama_cpp_batch_size: int = 128
    # Speed optimizations
    llama_cpp_spec_type: str = ""         # "ngram-mod", "ngram-simple" — speculative decoding (1.5-2x)
    llama_cpp_draft_max: int = 64         # max draft tokens for speculation
    llama_cpp_expert_offload: bool = False # offload MoE experts to CPU (-ot exps=CPU)
    llama_cpp_draft_model: str = ""       # path to draft GGUF for speculative decoding
    llama_cpp_lookup_cache: bool = False   # prompt lookup decoding (2-4x on code edits)
    kv_cache_type: str = "q8_0"           # KV cache quantization: q8_0, q4_0, f16
    temperature: float = 0.7  # Gemma 4 recommended (Unsloth: 1.0, we use 0.7 for coding focus)
    max_context_chars: int = 40000
    request_timeout_seconds: int = 120
    max_retries: int = 2


@dataclass(slots=True)
class SearchConfig:
    provider: str = "duckduckgo"
    google_api_key: str = ""
    google_cx: str = ""
    brave_api_key: str = ""
    serpapi_api_key: str = ""


@dataclass(slots=True)
class BrowserConfig:
    enabled: bool = True
    mcp_server_name: str = "browser"
    launch_command: str = "npx"
    launch_args: list[str] | None = None


@dataclass(slots=True)
class VoiceConfig:
    stt_provider: str = "whisper.cpp"
    tts_provider: str = "piper"
    whisper_model_path: str = ""
    faster_whisper_model: str = "small"
    kokoro_voice: str = "af_heart"
    piper_model_path: str = ""


@dataclass(slots=True)
class UIConfig:
    show_debug: bool = False
    thinking_mode: str = "full"


@dataclass(slots=True)
class AppConfig:
    runtime: RuntimeConfig
    search: SearchConfig
    browser: BrowserConfig
    voice: VoiceConfig
    ui: UIConfig


def get_home_dir() -> Path:
    override = os.environ.get("GEM_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".gem"


def ensure_home_dirs() -> Path:
    home = get_home_dir()
    for child in ("logs", "sessions", "jobs", "skills", "plugins", "audio"):
        (home / child).mkdir(parents=True, exist_ok=True)
    return home


def get_config_path() -> Path:
    return get_home_dir() / "config.toml"


def init_config_file() -> Path:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(DEFAULT_CONFIG)
    return path


def save_config(config: AppConfig) -> Path:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "[runtime]\n"
        f'provider = "{config.runtime.provider}"\n'
        f'base_url = "{config.runtime.base_url}"\n'
        f'profile = "{config.runtime.profile}"\n'
        f'model = "{config.runtime.model}"\n'
        f'mode = "{config.runtime.mode}"\n'
        f'planner_model = "{config.runtime.planner_model}"\n'
        f'draft_model = "{config.runtime.draft_model}"\n'
        f"planner_enabled = {'true' if config.runtime.planner_enabled else 'false'}\n"
        f"adaptive_execution = {'true' if config.runtime.adaptive_execution else 'false'}\n"
        f"escalation_enabled = {'true' if config.runtime.escalation_enabled else 'false'}\n"
        f'huggingface_model_id = "{config.runtime.huggingface_model_id}"\n'
        f'huggingface_device = "{config.runtime.huggingface_device}"\n'
        f'huggingface_dtype = "{config.runtime.huggingface_dtype}"\n'
        f'mlx_model_id = "{config.runtime.mlx_model_id}"\n'
        f'quant_preset = "{config.runtime.quant_preset}"\n'
        f'cache_policy = "{config.runtime.cache_policy}"\n'
        f"rolling_window_messages = {config.runtime.rolling_window_messages}\n"
        f"llama_cpp_gpu_layers = {config.runtime.llama_cpp_gpu_layers}\n"
        f"llama_cpp_threads = {config.runtime.llama_cpp_threads}\n"
        f"llama_cpp_batch_size = {config.runtime.llama_cpp_batch_size}\n"
        f'llama_cpp_spec_type = "{config.runtime.llama_cpp_spec_type}"\n'
        f"llama_cpp_draft_max = {config.runtime.llama_cpp_draft_max}\n"
        f"llama_cpp_expert_offload = {'true' if config.runtime.llama_cpp_expert_offload else 'false'}\n"
        f'llama_cpp_draft_model = "{config.runtime.llama_cpp_draft_model}"\n'
        f"llama_cpp_lookup_cache = {'true' if config.runtime.llama_cpp_lookup_cache else 'false'}\n"
        f'kv_cache_type = "{config.runtime.kv_cache_type}"\n'
        f"temperature = {config.runtime.temperature}\n"
        f"max_context_chars = {config.runtime.max_context_chars}\n"
        f"request_timeout_seconds = {config.runtime.request_timeout_seconds}\n"
        f"max_retries = {config.runtime.max_retries}\n\n"
        "[search]\n"
        f'provider = "{config.search.provider}"\n'
        f'google_api_key = "{config.search.google_api_key}"\n'
        f'google_cx = "{config.search.google_cx}"\n'
        f'brave_api_key = "{config.search.brave_api_key}"\n'
        f'serpapi_api_key = "{config.search.serpapi_api_key}"\n\n'
        "[browser]\n"
        f"enabled = {'true' if config.browser.enabled else 'false'}\n"
        f'mcp_server_name = "{config.browser.mcp_server_name}"\n'
        f'launch_command = "{config.browser.launch_command}"\n'
        f"launch_args = [{', '.join(f'\"{arg}\"' for arg in (config.browser.launch_args or []))}]\n\n"
        "[voice]\n"
        f'stt_provider = "{config.voice.stt_provider}"\n'
        f'tts_provider = "{config.voice.tts_provider}"\n'
        f'whisper_model_path = "{config.voice.whisper_model_path}"\n'
        f'faster_whisper_model = "{config.voice.faster_whisper_model}"\n'
        f'kokoro_voice = "{config.voice.kokoro_voice}"\n'
        f'piper_model_path = "{config.voice.piper_model_path}"\n\n'
        "[ui]\n"
        f"show_debug = {'true' if config.ui.show_debug else 'false'}\n"
        f'thinking_mode = "{config.ui.thinking_mode}"\n'
    )
    path.write_text(content)
    return path


def load_config() -> AppConfig:
    ensure_home_dirs()
    path = get_config_path()
    if not path.exists():
        init_config_file()
    data = tomllib.loads(path.read_text())
    runtime_data = data.get("runtime", {})
    search_data = data.get("search", {})
    browser_data = data.get("browser", {})
    voice_data = data.get("voice", {})
    ui_data = data.get("ui", {})
    runtime = RuntimeConfig(
        provider=os.environ.get("GEM_PROVIDER", runtime_data.get("provider", "ollama")),
        base_url=os.environ.get("GEM_BASE_URL", runtime_data.get("base_url", "http://localhost:11434")),
        profile=os.environ.get("GEM_PROFILE", runtime_data.get("profile", "e4b")),
        model=os.environ.get("GEM_MODEL", runtime_data.get("model", "")),
        mode=os.environ.get("GEM_MODE", runtime_data.get("mode", "balanced")),
        planner_model=os.environ.get("GEM_PLANNER_MODEL", runtime_data.get("planner_model", "gemma4:e2b")),
        draft_model=os.environ.get("GEM_DRAFT_MODEL", runtime_data.get("draft_model", "gemma4:e2b")),
        planner_enabled=str(os.environ.get("GEM_PLANNER_ENABLED", runtime_data.get("planner_enabled", True))).lower() in {"1", "true", "yes", "on"},
        adaptive_execution=str(os.environ.get("GEM_ADAPTIVE_EXECUTION", runtime_data.get("adaptive_execution", True))).lower() in {"1", "true", "yes", "on"},
        escalation_enabled=str(os.environ.get("GEM_ESCALATION_ENABLED", runtime_data.get("escalation_enabled", True))).lower() in {"1", "true", "yes", "on"},
        huggingface_model_id=os.environ.get("GEM_HF_MODEL_ID", runtime_data.get("huggingface_model_id", "")),
        huggingface_device=os.environ.get("GEM_HF_DEVICE", runtime_data.get("huggingface_device", "auto")),
        huggingface_dtype=os.environ.get("GEM_HF_DTYPE", runtime_data.get("huggingface_dtype", "auto")),
        mlx_model_id=os.environ.get("GEM_MLX_MODEL_ID", runtime_data.get("mlx_model_id", "")),
        quant_preset=os.environ.get("GEM_QUANT_PRESET", runtime_data.get("quant_preset", "balanced")),
        cache_policy=os.environ.get("GEM_CACHE_POLICY", runtime_data.get("cache_policy", "adaptive")),
        rolling_window_messages=int(os.environ.get("GEM_ROLLING_WINDOW_MESSAGES", runtime_data.get("rolling_window_messages", 24))),
        llama_cpp_gpu_layers=int(os.environ.get("GEM_LLAMA_CPP_GPU_LAYERS", runtime_data.get("llama_cpp_gpu_layers", 0))),
        llama_cpp_threads=int(os.environ.get("GEM_LLAMA_CPP_THREADS", runtime_data.get("llama_cpp_threads", 8))),
        llama_cpp_batch_size=int(os.environ.get("GEM_LLAMA_CPP_BATCH_SIZE", runtime_data.get("llama_cpp_batch_size", 128))),
        llama_cpp_spec_type=os.environ.get("GEM_LLAMA_CPP_SPEC_TYPE", runtime_data.get("llama_cpp_spec_type", "")),
        llama_cpp_draft_max=int(os.environ.get("GEM_LLAMA_CPP_DRAFT_MAX", runtime_data.get("llama_cpp_draft_max", 64))),
        llama_cpp_expert_offload=str(os.environ.get("GEM_LLAMA_CPP_EXPERT_OFFLOAD", runtime_data.get("llama_cpp_expert_offload", False))).lower() in {"1", "true", "yes", "on"},
        llama_cpp_draft_model=os.environ.get("GEM_LLAMA_CPP_DRAFT_MODEL", runtime_data.get("llama_cpp_draft_model", "")),
        llama_cpp_lookup_cache=str(os.environ.get("GEM_LLAMA_CPP_LOOKUP_CACHE", runtime_data.get("llama_cpp_lookup_cache", False))).lower() in {"1", "true", "yes", "on"},
        kv_cache_type=os.environ.get("GEM_KV_CACHE_TYPE", runtime_data.get("kv_cache_type", "q8_0")),
        temperature=float(os.environ.get("GEM_TEMPERATURE", runtime_data.get("temperature", 0.2))),
        max_context_chars=int(os.environ.get("GEM_MAX_CONTEXT_CHARS", runtime_data.get("max_context_chars", 40000))),
        request_timeout_seconds=int(os.environ.get("GEM_REQUEST_TIMEOUT_SECONDS", runtime_data.get("request_timeout_seconds", 120))),
        max_retries=int(os.environ.get("GEM_MAX_RETRIES", runtime_data.get("max_retries", 2))),
    )
    search = SearchConfig(
        provider=os.environ.get("GEM_SEARCH_PROVIDER", search_data.get("provider", "duckduckgo")),
        google_api_key=os.environ.get("GEM_GOOGLE_API_KEY", search_data.get("google_api_key", "")),
        google_cx=os.environ.get("GEM_GOOGLE_CX", search_data.get("google_cx", "")),
        brave_api_key=os.environ.get("GEM_BRAVE_API_KEY", search_data.get("brave_api_key", "")),
        serpapi_api_key=os.environ.get("GEM_SERPAPI_API_KEY", search_data.get("serpapi_api_key", "")),
    )
    browser = BrowserConfig(
        enabled=str(os.environ.get("GEM_BROWSER_ENABLED", browser_data.get("enabled", True))).lower() in {"1", "true", "yes", "on"},
        mcp_server_name=os.environ.get("GEM_BROWSER_MCP_SERVER_NAME", browser_data.get("mcp_server_name", "browser")),
        launch_command=os.environ.get("GEM_BROWSER_LAUNCH_COMMAND", browser_data.get("launch_command", "npx")),
        launch_args=list(browser_data.get("launch_args", ["-y", "@playwright/mcp@latest"])),
    )
    voice = VoiceConfig(
        stt_provider=os.environ.get("GEM_STT_PROVIDER", voice_data.get("stt_provider", "whisper.cpp")),
        tts_provider=os.environ.get("GEM_TTS_PROVIDER", voice_data.get("tts_provider", default_tts_provider())),
        whisper_model_path=os.environ.get("GEM_WHISPER_MODEL_PATH", voice_data.get("whisper_model_path", "")),
        faster_whisper_model=os.environ.get("GEM_FASTER_WHISPER_MODEL", voice_data.get("faster_whisper_model", "small")),
        kokoro_voice=os.environ.get("GEM_KOKORO_VOICE", voice_data.get("kokoro_voice", "af_heart")),
        piper_model_path=os.environ.get("GEM_PIPER_MODEL_PATH", voice_data.get("piper_model_path", "")),
    )
    ui = UIConfig(
        show_debug=str(os.environ.get("GEM_SHOW_DEBUG", ui_data.get("show_debug", False))).lower() in {"1", "true", "yes", "on"},
        thinking_mode=os.environ.get("GEM_THINKING_MODE", ui_data.get("thinking_mode", "summary")),
    )
    return AppConfig(runtime=runtime, search=search, browser=browser, voice=voice, ui=ui)
