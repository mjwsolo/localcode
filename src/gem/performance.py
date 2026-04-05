from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

from .config import AppConfig, save_config
from .models import ModelProfile


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
    kv_cache_type: str = "q8_0"
    notes: list[str] = None  # type: ignore[assignment]


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


def recommend_preset(machine: MachineProfile, requested_mode: str | None = None) -> PerformancePreset:
    mode = requested_mode or ("fast" if machine.tier == "small" else "balanced")
    runtime_provider = "ollama"
    notes: list[str] = []
    quant_preset = "balanced"
    cache_policy = "adaptive"
    rolling_window_messages = 24
    llama_cpp_gpu_layers = 0
    llama_cpp_threads = max(2, min(machine.cpu_cores, 8))
    llama_cpp_batch_size = 128
    if machine.system == "darwin" and machine.has_gpu:
        runtime_provider = "mlx-local" if mode in {"fast", "balanced"} else "ollama"
        notes.append("Apple Silicon detected: prefer MLX-local quantized models when available.")
    if machine.tier in {"small", "medium"}:
        profile = "gemma4-e2b" if mode == "fast" else "gemma4-e4b"
        max_context_chars = 10000 if mode == "fast" else 18000
        planner_enabled = True
        adaptive_execution = True
        draft_model = "gemma4:e2b"
        planner_model = "gemma4:e2b"
        quant_preset = "smallest" if mode == "fast" else "balanced"
        cache_policy = "rolling"
        rolling_window_messages = 16 if mode == "fast" else 20
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
    kv_cache_type = "q8_0"

    # Small tier (≤16GB): aggressive memory savings
    if machine.tier == "small":
        kv_cache_type = "q4_0"  # aggressive KV compression to free memory
        expert_offload = True   # MoE experts to CPU, attention stays on GPU
        notes.append("KV cache q4_0 + expert offload for 16GB memory constraint.")

    # When using llama.cpp: enable n-gram speculation
    if runtime_provider == "llama_cpp":
        spec_type = "ngram-mod"
        draft_max = 64
        notes.append("N-gram speculative decoding enabled (1.5-2x speedup).")

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
        kv_cache_type=kv_cache_type,
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
    config.runtime.kv_cache_type = preset.kv_cache_type
    save_config(config)
    return config


def benchmark_report(config: AppConfig, requested_mode: str | None = None) -> tuple[MachineProfile, PerformancePreset]:
    machine = detect_machine_profile()
    preset = recommend_preset(machine, requested_mode)
    return machine, preset
