from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

# Serializes config writes. The setup worker thread and the model
# picker can both call save_config concurrently; without this two
# interleaved write_text calls could leave a half-written TOML.
_SAVE_LOCK = threading.Lock()


DEFAULT_CONFIG = """[runtime]
provider = "llama_cpp"
base_url = "http://localhost:8081"
profile = "gemma4-26b-laptop"
model = "gemma26b-iq3"
mode = "fast"
execution_engine = "unified"
planner_model = ""
draft_model = ""
planner_enabled = false
planner_hints_enabled = false
adaptive_execution = false
escalation_enabled = false
low_overhead_mode = true
laptop_26b_runtime_mode = "turbo"
internal_thinking_mode = "off"
quant_preset = "balanced"
cache_policy = "adaptive"
rolling_window_messages = 24
llama_cpp_gpu_layers = 0
llama_cpp_threads = 10
llama_cpp_batch_size = 2048
llama_cpp_spec_type = ""
llama_cpp_draft_max = 64
llama_cpp_expert_offload = false
llama_cpp_draft_model = ""
llama_cpp_lookup_cache = false
kv_cache_type_k = "q8_0"
kv_cache_type_v = "turbo4"
llama_cpp_binary = ""
vision_enabled = false
temperature = 1.0
max_context_chars = 200000
request_timeout_seconds = 600
max_retries = 3
max_rounds = 0
thinking_budget_tokens = 0

[search]
provider = "duckduckgo"
google_api_key = ""
google_cx = ""
brave_api_key = ""
serpapi_api_key = ""

# [browser] and [voice] sections removed during T0.9 purge.

[ui]
show_debug = false
thinking_mode = "full"

[safety]
confirm_destructive = true
confirm_installs = true
show_diff_before_apply = true
jail_to_project = true
max_fix_retries = 3
max_replan_attempts = 2
auto_approve_agent = false

[logging]
enabled = true
log_prompts = true
log_responses = true
max_days = 30
"""


@dataclass
class RuntimeConfig:
    provider: str = "llama_cpp"
    base_url: str = "http://localhost:8081"
    profile: str = "gemma4-26b-laptop"
    model: str = "gemma26b-iq3"
    mode: str = "fast"
    execution_engine: str = "unified"
    planner_model: str = ""
    draft_model: str = ""
    planner_enabled: bool = False
    planner_hints_enabled: bool = False
    adaptive_execution: bool = False
    escalation_enabled: bool = False
    low_overhead_mode: bool = True
    max_rounds: int = 0  # 0 = unlimited interactive loop; positive for batch/eval
    thinking_budget_tokens: int = 0  # 0 = catalog default; negative disables
    laptop_26b_runtime_mode: str = "turbo"
    internal_thinking_mode: str = "off"
    quant_preset: str = "balanced"
    cache_policy: str = "adaptive"
    rolling_window_messages: int = 24
    llama_cpp_gpu_layers: int = 0
    llama_cpp_threads: int = -1   # -1 = auto-detect from CPU cores at startup
    llama_cpp_batch_size: int = -1  # -1 = auto-detect from platform at startup
    # Speed optimizations
    llama_cpp_spec_type: str = ""         # "ngram-mod", "ngram-simple" — speculative decoding (1.5-2x)
    llama_cpp_draft_max: int = 64         # max draft tokens for speculation
    llama_cpp_expert_offload: bool = False # offload MoE experts to CPU (-ot exps=CPU)
    llama_cpp_draft_model: str = ""       # path to draft GGUF for speculative decoding
    llama_cpp_lookup_cache: bool = False   # prompt lookup decoding (2-4x on code edits)
    kv_cache_type_k: str = "q8_0"          # K cache type: q8_0, q4_0, f16, turbo2, turbo3, turbo4
    kv_cache_type_v: str = "turbo4"        # V cache type: turbo4 recommended (3.8x compression, +0.23% PPL)
    llama_cpp_cache_reuse: int = 256       # --cache-reuse N: reuse KV chunks across partial prefix matches (0 = off). Recovers prefix-cache hits after mid-context edits/compaction shift the tail; the stable system-prompt prefix is already reused automatically per slot.
    llama_cpp_binary: str = ""             # custom llama-server path (e.g. TurboQuant fork)
    model_dir: str = ""                    # directory where GGUFs download to (blank → ~/.local/share/localcode/models)
    # Vision toggle. Tracks whether the multimodal projector should be
    # loaded (--mmproj) alongside the text decoder. Persisted so turning
    # vision OFF just relaunches WITHOUT --mmproj (frees RAM) while KEEPING
    # the projector file on disk — re-enabling is instant, no re-download.
    vision_enabled: bool = False
    temperature: float = 1.0  # Unsloth's official Gemma 4 recommendation — prevents IQ3_S mode-collapse loops
    max_context_chars: int = 200000
    request_timeout_seconds: int = 600
    max_retries: int = 3


@dataclass
class SearchConfig:
    provider: str = "duckduckgo"
    google_api_key: str = ""
    google_cx: str = ""
    brave_api_key: str = ""
    serpapi_api_key: str = ""


# BrowserConfig / VoiceConfig removed during T0.9 — no UI path fed the
# underlying modules, the MCP browser server was unreachable, and the
# voice stack (whisper.cpp / kokoro / piper) had no active consumer.


@dataclass
class UIConfig:
    show_debug: bool = False
    thinking_mode: str = "full"
    sounds_enabled: bool = False  # afplay on turn-done + approval-needed


@dataclass
class SafetyConfig:
    confirm_destructive: bool = True
    confirm_installs: bool = True
    show_diff_before_apply: bool = True
    jail_to_project: bool = True
    max_fix_retries: int = 3
    max_replan_attempts: int = 2
    auto_approve_agent: bool = False  # auto-approve all tools in agent mode


@dataclass
class LoggingConfig:
    enabled: bool = True
    log_prompts: bool = True
    log_responses: bool = True
    max_days: int = 30


@dataclass
class AppConfig:
    runtime: RuntimeConfig
    search: SearchConfig
    ui: UIConfig
    safety: SafetyConfig = None  # type: ignore[assignment]
    logging: LoggingConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.safety is None:
            self.safety = SafetyConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


def get_home_dir() -> Path:
    override = os.environ.get("LOCALCODE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".localcode"


def ensure_home_dirs() -> Path:
    home = get_home_dir()
    for child in ("logs", "sessions", "jobs", "skills", "plugins"):
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
        f'execution_engine = "{config.runtime.execution_engine}"\n'
        f'planner_model = "{config.runtime.planner_model}"\n'
        f'draft_model = "{config.runtime.draft_model}"\n'
        f"planner_enabled = {'true' if config.runtime.planner_enabled else 'false'}\n"
        f"planner_hints_enabled = {'true' if config.runtime.planner_hints_enabled else 'false'}\n"
        f"adaptive_execution = {'true' if config.runtime.adaptive_execution else 'false'}\n"
        f"escalation_enabled = {'true' if config.runtime.escalation_enabled else 'false'}\n"
        f"low_overhead_mode = {'true' if config.runtime.low_overhead_mode else 'false'}\n"
        f"max_rounds = {config.runtime.max_rounds}\n"
        f"thinking_budget_tokens = {config.runtime.thinking_budget_tokens}\n"
        f'laptop_26b_runtime_mode = "{config.runtime.laptop_26b_runtime_mode}"\n'
        f'internal_thinking_mode = "{config.runtime.internal_thinking_mode}"\n'
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
        f'kv_cache_type_k = "{config.runtime.kv_cache_type_k}"\n'
        f'kv_cache_type_v = "{config.runtime.kv_cache_type_v}"\n'
        f'llama_cpp_binary = "{config.runtime.llama_cpp_binary}"\n'
        f"llama_cpp_cache_reuse = {config.runtime.llama_cpp_cache_reuse}\n"
        f'model_dir = "{config.runtime.model_dir}"\n'
        f"vision_enabled = {'true' if config.runtime.vision_enabled else 'false'}\n"
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
        # [browser] / [voice] sections removed during T0.9 purge.
        "[ui]\n"
        f"show_debug = {'true' if config.ui.show_debug else 'false'}\n"
        f'thinking_mode = "{config.ui.thinking_mode}"\n'
        f"sounds_enabled = {'true' if config.ui.sounds_enabled else 'false'}\n\n"
        "[safety]\n"
        f"confirm_destructive = {'true' if config.safety.confirm_destructive else 'false'}\n"
        f"confirm_installs = {'true' if config.safety.confirm_installs else 'false'}\n"
        f"show_diff_before_apply = {'true' if config.safety.show_diff_before_apply else 'false'}\n"
        f"jail_to_project = {'true' if config.safety.jail_to_project else 'false'}\n"
        f"max_fix_retries = {config.safety.max_fix_retries}\n"
        f"max_replan_attempts = {config.safety.max_replan_attempts}\n"
        f"auto_approve_agent = {'true' if config.safety.auto_approve_agent else 'false'}\n\n"
        "[logging]\n"
        f"enabled = {'true' if config.logging.enabled else 'false'}\n"
        f"log_prompts = {'true' if config.logging.log_prompts else 'false'}\n"
        f"log_responses = {'true' if config.logging.log_responses else 'false'}\n"
        f"max_days = {config.logging.max_days}\n"
    )
    # Atomic write: a reader (or a crash) must never observe a
    # half-written config. Write to a temp file in the same directory,
    # then os.replace (atomic rename on the same filesystem). The lock
    # serializes concurrent savers so their temp files don't race on the
    # final rename.
    with _SAVE_LOCK:
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        tmp.write_text(content)
        os.replace(tmp, path)
    return path


def load_config() -> AppConfig:
    ensure_home_dirs()
    path = get_config_path()
    if not path.exists():
        init_config_file()
    data = tomllib.loads(path.read_text())
    runtime_data = data.get("runtime", {})
    search_data = data.get("search", {})
    # browser / voice sections removed during T0.9 purge
    ui_data = data.get("ui", {})
    runtime = RuntimeConfig(
        provider=os.environ.get("LOCALCODE_PROVIDER", runtime_data.get("provider", "llama_cpp")),
        base_url=os.environ.get("LOCALCODE_BASE_URL", runtime_data.get("base_url", "http://localhost:8081")),
        profile=os.environ.get("LOCALCODE_PROFILE", runtime_data.get("profile", "e4b")),
        model=os.environ.get("LOCALCODE_MODEL", runtime_data.get("model", "")),
        mode=os.environ.get("LOCALCODE_MODE", runtime_data.get("mode", "balanced")),
        execution_engine=os.environ.get("LOCALCODE_EXECUTION_ENGINE", runtime_data.get("execution_engine", "unified")),
        planner_model=os.environ.get("LOCALCODE_PLANNER_MODEL", runtime_data.get("planner_model", "gemma4:e2b")),
        draft_model=os.environ.get("LOCALCODE_DRAFT_MODEL", runtime_data.get("draft_model", "gemma4:e2b")),
        planner_enabled=str(os.environ.get("LOCALCODE_PLANNER_ENABLED", runtime_data.get("planner_enabled", True))).lower() in {"1", "true", "yes", "on"},
        planner_hints_enabled=str(os.environ.get("LOCALCODE_PLANNER_HINTS_ENABLED", runtime_data.get("planner_hints_enabled", False))).lower() in {"1", "true", "yes", "on"},
        adaptive_execution=str(os.environ.get("LOCALCODE_ADAPTIVE_EXECUTION", runtime_data.get("adaptive_execution", True))).lower() in {"1", "true", "yes", "on"},
        escalation_enabled=str(os.environ.get("LOCALCODE_ESCALATION_ENABLED", runtime_data.get("escalation_enabled", True))).lower() in {"1", "true", "yes", "on"},
        low_overhead_mode=str(os.environ.get("LOCALCODE_LOW_OVERHEAD_MODE", runtime_data.get("low_overhead_mode", False))).lower() in {"1", "true", "yes", "on"},
        max_rounds=max(0, int(os.environ.get("LOCALCODE_MAX_ROUNDS", runtime_data.get("max_rounds", runtime_data.get("max_turns", 0))))),
        thinking_budget_tokens=int(os.environ.get("LOCALCODE_THINKING_BUDGET", runtime_data.get("thinking_budget_tokens", 0))),
        laptop_26b_runtime_mode=os.environ.get("LOCALCODE_LAPTOP_26B_RUNTIME_MODE", runtime_data.get("laptop_26b_runtime_mode", "auto")),
        internal_thinking_mode=os.environ.get("LOCALCODE_INTERNAL_THINKING_MODE", runtime_data.get("internal_thinking_mode", "off")),
        quant_preset=os.environ.get("LOCALCODE_QUANT_PRESET", runtime_data.get("quant_preset", "balanced")),
        cache_policy=os.environ.get("LOCALCODE_CACHE_POLICY", runtime_data.get("cache_policy", "adaptive")),
        rolling_window_messages=int(os.environ.get("LOCALCODE_ROLLING_WINDOW_MESSAGES", runtime_data.get("rolling_window_messages", 24))),
        llama_cpp_gpu_layers=int(os.environ.get("LOCALCODE_LLAMA_CPP_GPU_LAYERS", runtime_data.get("llama_cpp_gpu_layers", 0))),
        llama_cpp_threads=int(os.environ.get("LOCALCODE_LLAMA_CPP_THREADS", runtime_data.get("llama_cpp_threads", 8))),
        llama_cpp_batch_size=int(os.environ.get("LOCALCODE_LLAMA_CPP_BATCH_SIZE", runtime_data.get("llama_cpp_batch_size", 128))),
        # These were saved but never loaded back (silent data loss on every
        # save→restart cycle): custom model dir and cache-reuse tuning.
        llama_cpp_cache_reuse=int(os.environ.get("LOCALCODE_LLAMA_CPP_CACHE_REUSE", runtime_data.get("llama_cpp_cache_reuse", 256))),
        model_dir=os.environ.get("LOCALCODE_MODEL_DIR", runtime_data.get("model_dir", "")),
        vision_enabled=str(os.environ.get("LOCALCODE_VISION_ENABLED", runtime_data.get("vision_enabled", False))).lower() in {"1", "true", "yes", "on"},
        # `llama_cpp_spec_type` migration (2026-04-26): old configs
        # auto-saved "ngram-mod" or "ngram-simple" from a benchmark
        # preset. N-gram speculative decoding looks up matching
        # n-grams in the recent output and feeds them as draft tokens
        # — same class of bug as --lookup-cache-dynamic. When the
        # model emits a sentence once, the n-gram lookup finds it
        # and proposes it as the next-token draft, which the model
        # accepts → infinite verbatim repetition ("Actually, I'll
        # just use write_file." × ∞ in screenshot 2026-04-26).
        # Force-disable on load even if saved file says ngram-*,
        # unless the user EXPLICITLY opts in via env var.
        llama_cpp_spec_type=os.environ.get("LOCALCODE_LLAMA_CPP_SPEC_TYPE", ""),
        llama_cpp_draft_max=int(os.environ.get("LOCALCODE_LLAMA_CPP_DRAFT_MAX", runtime_data.get("llama_cpp_draft_max", 64))),
        llama_cpp_expert_offload=str(os.environ.get("LOCALCODE_LLAMA_CPP_EXPERT_OFFLOAD", runtime_data.get("llama_cpp_expert_offload", False))).lower() in {"1", "true", "yes", "on"},
        llama_cpp_draft_model=os.environ.get("LOCALCODE_LLAMA_CPP_DRAFT_MODEL", runtime_data.get("llama_cpp_draft_model", "")),
        # `llama_cpp_lookup_cache` migration (2026-04-26): old configs
        # auto-saved this as `true` from a previous performance.py
        # preset. Prompt-lookup speculative decoding causes verbatim
        # repetition on chat workloads (model copies prior in-context
        # answers to repeated questions; documented in commit 6d13f3d).
        # Force-disable when loading even if the saved file says true,
        # unless the user EXPLICITLY opts in via the env var. Saved
        # config gets rewritten to false on the next save_config call.
        llama_cpp_lookup_cache=str(os.environ.get("LOCALCODE_LLAMA_CPP_LOOKUP_CACHE", "false")).lower() in {"1", "true", "yes", "on"},
        kv_cache_type_k=os.environ.get("LOCALCODE_KV_CACHE_TYPE_K", runtime_data.get("kv_cache_type_k", runtime_data.get("kv_cache_type", "q8_0"))),
        kv_cache_type_v=os.environ.get("LOCALCODE_KV_CACHE_TYPE_V", runtime_data.get("kv_cache_type_v", runtime_data.get("kv_cache_type", "turbo4"))),
        llama_cpp_binary=os.environ.get("LOCALCODE_LLAMA_CPP_BINARY", runtime_data.get("llama_cpp_binary", "")),
        temperature=float(os.environ.get("LOCALCODE_TEMPERATURE", runtime_data.get("temperature", 1.0))),
        max_context_chars=int(os.environ.get("LOCALCODE_MAX_CONTEXT_CHARS", runtime_data.get("max_context_chars", 40000))),
        request_timeout_seconds=int(os.environ.get("LOCALCODE_REQUEST_TIMEOUT_SECONDS", runtime_data.get("request_timeout_seconds", 120))),
        max_retries=int(os.environ.get("LOCALCODE_MAX_RETRIES", runtime_data.get("max_retries", 2))),
    )
    search = SearchConfig(
        provider=os.environ.get("LOCALCODE_SEARCH_PROVIDER", search_data.get("provider", "duckduckgo")),
        google_api_key=os.environ.get("LOCALCODE_GOOGLE_API_KEY", search_data.get("google_api_key", "")),
        google_cx=os.environ.get("LOCALCODE_GOOGLE_CX", search_data.get("google_cx", "")),
        brave_api_key=os.environ.get("LOCALCODE_BRAVE_API_KEY", search_data.get("brave_api_key", "")),
        serpapi_api_key=os.environ.get("LOCALCODE_SERPAPI_API_KEY", search_data.get("serpapi_api_key", "")),
    )
    # browser / voice config loaders removed during T0.9 purge.
    ui = UIConfig(
        show_debug=str(os.environ.get("LOCALCODE_SHOW_DEBUG", ui_data.get("show_debug", False))).lower() in {"1", "true", "yes", "on"},
        thinking_mode=os.environ.get("LOCALCODE_THINKING_MODE", ui_data.get("thinking_mode", "summary")),
        # Was saved by save_config but never loaded here — the /sounds toggle
        # silently reset to off on every restart.
        sounds_enabled=str(ui_data.get("sounds_enabled", False)).lower() in {"1", "true", "yes", "on"},
    )
    safety_data = data.get("safety", {})
    safety = SafetyConfig(
        confirm_destructive=str(safety_data.get("confirm_destructive", True)).lower() in {"1", "true", "yes", "on"},
        confirm_installs=str(safety_data.get("confirm_installs", True)).lower() in {"1", "true", "yes", "on"},
        show_diff_before_apply=str(safety_data.get("show_diff_before_apply", True)).lower() in {"1", "true", "yes", "on"},
        jail_to_project=str(safety_data.get("jail_to_project", True)).lower() in {"1", "true", "yes", "on"},
        max_fix_retries=int(safety_data.get("max_fix_retries", 3)),
        max_replan_attempts=int(safety_data.get("max_replan_attempts", 2)),
        auto_approve_agent=str(safety_data.get("auto_approve_agent", False)).lower() in {"1", "true", "yes", "on"},
    )
    logging_data = data.get("logging", {})
    logging_cfg = LoggingConfig(
        enabled=str(logging_data.get("enabled", True)).lower() in {"1", "true", "yes", "on"},
        log_prompts=str(logging_data.get("log_prompts", True)).lower() in {"1", "true", "yes", "on"},
        log_responses=str(logging_data.get("log_responses", True)).lower() in {"1", "true", "yes", "on"},
        max_days=int(logging_data.get("max_days", 30)),
    )
    # Migration: the mlx-local, huggingface-local, AND ollama backends were
    # removed — llama_cpp (the tuned llama-server) is the sole HTTP runtime.
    # A persisted `provider` pointing at any removed backend is coerced to
    # llama_cpp, and a stale 11434 (ollama) base_url is moved to the llama_cpp
    # port so URLs are correct. (Ollama was the slow ~1 tok/s fallback; users
    # now get the fast path or a clear "start the server" error, not a silent
    # 50× slowdown.)
    if runtime.provider in ("mlx-local", "huggingface-local", "ollama"):
        runtime.provider = "llama_cpp"
        if not runtime.base_url or "11434" in runtime.base_url:
            runtime.base_url = "http://localhost:8081"

    config = AppConfig(runtime=runtime, search=search,
                       ui=ui, safety=safety, logging=logging_cfg)

    # Layer project-level config on top (if .localcode/config.toml exists)
    config = _apply_project_config(config)
    return config


def _apply_project_config(config: AppConfig, project_root: Path | None = None) -> AppConfig:
    """Layer project-level .localcode/config.toml on top of global config.

    Project config can override: runtime.mode, runtime.temperature,
    safety settings, and ui settings. NOT provider/model (those are global).
    """
    if project_root is None:
        project_root = Path.cwd()

    project_config_path = project_root / ".localcode" / "config.toml"
    if not project_config_path.is_file():
        return config

    try:
        data = tomllib.loads(project_config_path.read_text())
    except Exception:
        return config

    # Override allowed runtime fields
    rt = data.get("runtime", {})
    if "mode" in rt:
        config.runtime.mode = rt["mode"]
    if "execution_engine" in rt:
        config.runtime.execution_engine = str(rt["execution_engine"])
    if "temperature" in rt:
        config.runtime.temperature = float(rt["temperature"])
    if "max_context_chars" in rt:
        config.runtime.max_context_chars = int(rt["max_context_chars"])
    if "low_overhead_mode" in rt:
        config.runtime.low_overhead_mode = str(rt["low_overhead_mode"]).lower() in {"1", "true", "yes", "on"}
    if "laptop_26b_runtime_mode" in rt:
        config.runtime.laptop_26b_runtime_mode = str(rt["laptop_26b_runtime_mode"]).strip() or "auto"
    if "internal_thinking_mode" in rt:
        config.runtime.internal_thinking_mode = str(rt["internal_thinking_mode"]).strip() or "off"
    if "max_rounds" in rt:
        config.runtime.max_rounds = max(0, int(rt["max_rounds"]))
    if "thinking_budget_tokens" in rt:
        config.runtime.thinking_budget_tokens = int(rt["thinking_budget_tokens"])
    if "planner_hints_enabled" in rt:
        config.runtime.planner_hints_enabled = str(rt["planner_hints_enabled"]).lower() in {"1", "true", "yes", "on"}

    # Override safety
    sf = data.get("safety", {})
    for field_name in ("confirm_destructive", "confirm_installs", "show_diff_before_apply",
                       "jail_to_project", "auto_approve_agent"):
        if field_name in sf:
            setattr(config.safety, field_name, str(sf[field_name]).lower() in {"1", "true", "yes", "on"})
    for field_name in ("max_fix_retries", "max_replan_attempts"):
        if field_name in sf:
            setattr(config.safety, field_name, int(sf[field_name]))

    # Override UI
    ui = data.get("ui", {})
    if "thinking_mode" in ui:
        config.ui.thinking_mode = ui["thinking_mode"]
    if "show_debug" in ui:
        config.ui.show_debug = str(ui["show_debug"]).lower() in {"1", "true", "yes", "on"}
    if "sounds_enabled" in ui:
        config.ui.sounds_enabled = str(ui["sounds_enabled"]).lower() in {"1", "true", "yes", "on"}

    return config
