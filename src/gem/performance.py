from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

import json

from .config import AppConfig, ensure_home_dirs, save_config


@dataclass(slots=True)
class MachineProfile:
    system: str
    cpu_cores: int
    memory_gb: int
    gpu_summary: str
    has_gpu: bool
    tier: str


@dataclass(slots=True)
class PerformancePreset:
    mode: str
    runtime_provider: str
    profile: str
    max_context_chars: int
    planner_enabled: bool
    adaptive_execution: bool
    planner_model: str
    draft_model: str
    quant_preset: str
    cache_policy: str
    rolling_window_messages: int
    llama_cpp_gpu_layers: int
    llama_cpp_threads: int
    llama_cpp_batch_size: int
    # Speed optimization fields
    llama_cpp_spec_type: str = ""
    llama_cpp_draft_max: int = 64
    llama_cpp_expert_offload: bool = False
    llama_cpp_draft_model: str = ""
    llama_cpp_lookup_cache: bool = False
    kv_cache_type_k: str = "q8_0"
    kv_cache_type_v: str = "turbo4"
    low_overhead_mode: bool = False
    laptop_26b_runtime_mode: str = "speed"
    notes: list[str] = None  # type: ignore[assignment]


def should_promote_legacy_default_to_laptop_26b(config: AppConfig, machine: MachineProfile) -> bool:
    profile = (config.runtime.profile or "").strip().lower()
    model = (config.runtime.model or "").strip()
    if model:
        return False
    if profile not in {"", "e4b", "gemma4-e4b"}:
        return False
    return machine.system == "darwin" and machine.has_gpu and machine.tier in {"small", "medium"}


def resolve_laptop_26b_runtime_mode(config: AppConfig, machine: MachineProfile) -> str:
    requested = (config.runtime.laptop_26b_runtime_mode or "auto").strip().lower()
    if requested in {"speed", "fit"}:
        return requested
    if machine.system != "darwin" or not machine.has_gpu:
        return "fit"

    explicit_model = (config.runtime.model or "").strip().lower()
    explicit_provider = (config.runtime.provider or "").strip().lower()
    if explicit_provider == "llama_cpp":
        return "fit"
    if explicit_provider == "ollama" and explicit_model and not explicit_model.endswith(".gguf"):
        return "speed"
    if explicit_model.endswith(".gguf") or any(token in explicit_model for token in ("iq", "q2_", "q3_", "q4_", "q5_", "q6_", "q8_")):
        return "fit"
    telemetry = _load_runtime_telemetry()
    speed_stats = telemetry.get("speed", {})
    fit_stats = telemetry.get("fit", {})
    if speed_stats.get("samples", 0) >= 2 and fit_stats.get("samples", 0) >= 2:
        speed_first = float(speed_stats.get("ema_first_token_s", 0.0) or 0.0)
        fit_first = float(fit_stats.get("ema_first_token_s", 0.0) or 0.0)
        if speed_first and fit_first:
            return "fit" if fit_first <= speed_first * 0.9 else "speed"
    if shutil.which("ollama") is not None:
        return "speed"
    if shutil.which("llama-server") is not None or shutil.which("llama_cpp.server") is not None:
        return "fit"
    return "speed"


def _load_runtime_telemetry() -> dict[str, dict]:
    path = ensure_home_dirs() / "memory.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    telemetry = data.get("runtime_telemetry", {})
    bucket = telemetry.get("gemma4-26b-laptop", {})
    return bucket if isinstance(bucket, dict) else {}


def _run_capture(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return (result.stdout or result.stderr or "").strip()


def detect_machine_profile() -> MachineProfile:
    system = platform.system().lower()
    cpu_cores = os.cpu_count() or 4
    memory_gb = _detect_memory_gb(system)
    gpu_summary = _detect_gpu_summary(system)
    has_gpu = gpu_summary not in {"", "none"}
    if memory_gb <= 16:
        tier = "small"
    elif memory_gb <= 32:
        tier = "medium"
    else:
        tier = "large"
    if has_gpu and memory_gb >= 48:
        tier = "workstation"
    return MachineProfile(
        system=system,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        gpu_summary=gpu_summary or "none",
        has_gpu=has_gpu,
        tier=tier,
    )


def _detect_memory_gb(system: str) -> int:
    if system == "darwin":
        raw = _run_capture(["sysctl", "-n", "hw.memsize"])
        if raw.isdigit():
            return max(1, int(raw) // (1024 ** 3))
    if system == "linux":
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            text = meminfo.read_text(errors="replace")
            match = re.search(r"MemTotal:\s+(\d+)\s+kB", text)
            if match:
                return max(1, int(match.group(1)) // (1024 ** 2))
    return 16


def _detect_gpu_summary(system: str) -> str:
    if shutil.which("nvidia-smi"):
        output = _run_capture(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if output:
            return output.splitlines()[0]
    if system == "darwin":
        output = _run_capture(["system_profiler", "SPDisplaysDataType"])
        match = re.search(r"Chipset Model:\s+(.+)", output)
        if match:
            return match.group(1).strip()
    return ""


def recommend_preset(
    machine: MachineProfile,
    requested_mode: str | None = None,
    laptop_26b_runtime_mode: str = "auto",
) -> PerformancePreset:
    mode = requested_mode or ("fast" if machine.tier == "small" else "balanced")
    runtime_provider = "ollama"
    notes: list[str] = []
    quant_preset = "balanced"
    cache_policy = "adaptive"
    rolling_window_messages = 24
    llama_cpp_gpu_layers = 0
    llama_cpp_threads = max(2, min(machine.cpu_cores, 8))
    llama_cpp_batch_size = 128
    if str(laptop_26b_runtime_mode).lower() in {"speed", "fit"}:
        laptop_runtime_mode = str(laptop_26b_runtime_mode).lower()
    else:
        laptop_runtime_mode = "speed"
    if machine.system == "darwin" and machine.has_gpu:
        runtime_provider = "ollama"
        notes.append("Apple Silicon detected: prefer MLX-local quantized models when available.")
    if machine.tier in {"small", "medium"}:
        use_laptop_26b = machine.system == "darwin" and machine.has_gpu
        profile = "gemma4-26b-laptop" if use_laptop_26b else ("gemma4-e2b" if mode == "fast" else "gemma4-e4b")
        max_context_chars = 10000 if use_laptop_26b else (10000 if mode == "fast" else 18000)
        planner_enabled = False if use_laptop_26b else True
        adaptive_execution = False if use_laptop_26b else True
        draft_model = ""
        planner_model = ""
        quant_preset = "fastest" if use_laptop_26b else ("smallest" if mode == "fast" else "balanced")
        cache_policy = "rolling"
        rolling_window_messages = 12 if use_laptop_26b else (16 if mode == "fast" else 20)
        if use_laptop_26b:
            notes.append("Disciplined 26B A4B laptop mode: single-model path, tight context, early compaction, low-overhead runtime.")
            notes.append(
                "Runtime preference: "
                + ("speed-first MLX/Ollama path." if laptop_runtime_mode == "speed" else "fit-first llama.cpp mmap path.")
            )
        else:
            draft_model = "gemma4:e2b"
            planner_model = "gemma4:e2b"
            notes.append("Prefer quantized small variants and early compaction.")
    elif machine.tier == "large":
        profile = "gemma4-e4b" if mode == "fast" else "gemma4-26b-moe"
        max_context_chars = 22000 if mode == "fast" else 36000
        planner_enabled = True
        adaptive_execution = True
        draft_model = "gemma4:e4b"
        planner_model = "gemma4:e2b"
        quant_preset = "fastest" if mode == "fast" else "balanced"
        cache_policy = "adaptive"
        rolling_window_messages = 24 if mode == "fast" else 32
        notes.append("Use planner routing before large-model synthesis.")
    else:
        profile = "gemma4-26b-moe" if mode != "deep" else "gemma4-31b"
        max_context_chars = 42000 if mode != "deep" else 64000
        planner_enabled = True
        adaptive_execution = True
        draft_model = "gemma4:e4b"
        planner_model = "gemma4:e2b"
        quant_preset = "balanced" if mode != "deep" else "best"
        cache_policy = "adaptive"
        rolling_window_messages = 32 if mode != "deep" else 48
        notes.append("Large hardware can sustain deeper agent loops and larger context.")
    if machine.system != "darwin" and machine.has_gpu:
        runtime_provider = "llama_cpp" if mode == "fast" else "ollama"
        if runtime_provider == "llama_cpp":
            llama_cpp_gpu_layers = 999
            llama_cpp_threads = max(4, min(machine.cpu_cores, 12))
            llama_cpp_batch_size = 256 if machine.tier in {"large", "workstation"} else 128
            notes.append("llama.cpp server mode is recommended for aggressive low-latency tuning.")
    if mode == "deep":
        max_context_chars = max(max_context_chars, 32000)

    # ── Speed optimization defaults per tier ──
    spec_type = ""
    draft_max = 64
    expert_offload = False
    llama_cpp_draft_model_path = ""
    lookup_cache = False
    kv_cache_type_k = "q8_0"
    kv_cache_type_v = "turbo4"

    # Small tier (≤16GB): TurboQuant KV compression + CPU mmap
    if machine.tier == "small":
        kv_cache_type_k = "q8_0"   # preserve key quality
        kv_cache_type_v = "turbo4"  # 3.8x V compression, +0.23% PPL — avoids pre-M5 turbo3 regression
        # Note: expert_offload with -ngl 999 causes OOM on 16GB.
        # Use -ngl 0 (pure CPU + mmap) which is stable and fast enough.
        expert_offload = False
        llama_cpp_gpu_layers = 0  # CPU mmap is more stable on 16GB
        notes.append("TurboQuant KV (q8_0-K + turbo4-V) + CPU mmap for 16GB.")

    # Apple Silicon + 26B laptop: always use llama.cpp with TurboQuant
    # MLX 4-bit doesn't fit on 16GB, Ollama doesn't support TurboQuant KV cache
    if machine.system == "darwin" and machine.has_gpu and profile == "gemma4-26b-laptop":
        runtime_provider = "llama_cpp"
        llama_cpp_gpu_layers = 0
        lookup_cache = True
        notes.append("TurboQuant llama.cpp for 26B laptop mode (MLX/Ollama can't fit or lack turbo KV).")
    elif machine.system == "darwin" and machine.has_gpu and profile == "gemma4-26b-moe":
        runtime_provider = "ollama"
        notes.append("Ollama MLX backend preferred for MoE models on Apple Silicon (3x faster).")

    # When using llama.cpp: enable n-gram speculation + prompt lookup
    if runtime_provider == "llama_cpp":
        spec_type = "ngram-mod"
        draft_max = 64
        lookup_cache = True  # prompt lookup decoding (2-4x on code edits)
        notes.append("N-gram speculative decoding + prompt lookup enabled.")

    return PerformancePreset(
        mode=mode,
        runtime_provider=runtime_provider,
        profile=profile,
        max_context_chars=max_context_chars,
        planner_enabled=planner_enabled,
        adaptive_execution=adaptive_execution,
        planner_model=planner_model,
        draft_model=draft_model,
        quant_preset=quant_preset,
        cache_policy=cache_policy,
        rolling_window_messages=rolling_window_messages,
        llama_cpp_gpu_layers=llama_cpp_gpu_layers,
        llama_cpp_threads=llama_cpp_threads,
        llama_cpp_batch_size=llama_cpp_batch_size,
        llama_cpp_spec_type=spec_type,
        llama_cpp_draft_max=draft_max,
        llama_cpp_expert_offload=expert_offload,
        llama_cpp_draft_model=llama_cpp_draft_model_path,
        llama_cpp_lookup_cache=lookup_cache,
        kv_cache_type_k=kv_cache_type_k,
        kv_cache_type_v=kv_cache_type_v,
        low_overhead_mode=(profile == "gemma4-26b-laptop"),
        laptop_26b_runtime_mode=laptop_runtime_mode,
        notes=notes,
    )


def apply_preset(config: AppConfig, preset: PerformancePreset, model: str | None = None) -> AppConfig:
    config.runtime.provider = preset.runtime_provider
    config.runtime.profile = preset.profile
    if model:
        config.runtime.model = model
    config.runtime.max_context_chars = preset.max_context_chars
    config.runtime.mode = preset.mode
    config.runtime.planner_enabled = preset.planner_enabled
    config.runtime.adaptive_execution = preset.adaptive_execution
    config.runtime.planner_model = preset.planner_model
    config.runtime.draft_model = preset.draft_model
    config.runtime.quant_preset = preset.quant_preset
    config.runtime.cache_policy = preset.cache_policy
    config.runtime.rolling_window_messages = preset.rolling_window_messages
    config.runtime.llama_cpp_gpu_layers = preset.llama_cpp_gpu_layers
    config.runtime.llama_cpp_threads = preset.llama_cpp_threads
    config.runtime.llama_cpp_batch_size = preset.llama_cpp_batch_size
    config.runtime.llama_cpp_spec_type = preset.llama_cpp_spec_type
    config.runtime.llama_cpp_draft_max = preset.llama_cpp_draft_max
    config.runtime.llama_cpp_expert_offload = preset.llama_cpp_expert_offload
    config.runtime.llama_cpp_draft_model = preset.llama_cpp_draft_model
    config.runtime.llama_cpp_lookup_cache = preset.llama_cpp_lookup_cache
    config.runtime.kv_cache_type_k = preset.kv_cache_type_k
    config.runtime.kv_cache_type_v = preset.kv_cache_type_v
    config.runtime.low_overhead_mode = preset.low_overhead_mode
    config.runtime.laptop_26b_runtime_mode = preset.laptop_26b_runtime_mode
    config.runtime.execution_engine = "unified"
    save_config(config)
    return config


def benchmark_report(config: AppConfig, requested_mode: str | None = None) -> tuple[MachineProfile, PerformancePreset]:
    machine = detect_machine_profile()
    laptop_mode = resolve_laptop_26b_runtime_mode(config, machine)
    preset = recommend_preset(machine, requested_mode, laptop_mode)
    return machine, preset
