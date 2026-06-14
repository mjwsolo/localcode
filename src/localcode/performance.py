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


def metal_gpu_available(memory_gb: int | None = None) -> bool:
    """Check if Metal GPU is usable for full model offload RIGHT NOW.

    On 32GB+ Macs: always True (Metal working set is large enough).
    On 16GB Macs: only True if iogpu.wired_limit_mb >= 14000.
    On <16GB or non-Mac: always False.

    This is the SINGLE SOURCE OF TRUTH. Never assume GPU works just
    because the config says ngl=999.
    """
    if platform.system().lower() != "darwin":
        return False
    if memory_gb is None:
        try:
            mem_bytes = int(subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip())
            memory_gb = mem_bytes // (1024 ** 3)
        except Exception:
            memory_gb = 16
    if memory_gb >= 32:
        return True  # 32GB+ never needs sysctl
    if memory_gb < 16:
        return False  # 8GB can't fit model on GPU
    # 16GB: must check sysctl
    try:
        r = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True, text=True, timeout=2,
        )
        return int(r.stdout.strip()) >= 12000
    except Exception:
        return False


_LAUNCHDAEMON_PLIST = "/Library/LaunchDaemons/com.localcode.gpu-unlock.plist"
_PLIST_CONTENT = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.localcode.gpu-unlock</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/sbin/sysctl</string>
        <string>iogpu.wired_limit_mb=14336</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
"""


def _sudo_via_osascript(shell_cmd: str) -> bool:
    """Run a shell command as root using macOS native password dialog.

    Uses osascript to show the standard macOS auth prompt — works from
    any context (TUI, CLI, background process, no TTY needed).
    The dialog shows "LocalCode" and explains why it needs permission.
    """
    escaped = shell_cmd.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'do shell script "{escaped}" '
        f'with administrator privileges '
        f'with prompt "LocalCode needs to increase GPU memory to run the AI model. This is a one-time setup."'
    )
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


def ensure_gpu_unlock() -> bool:
    """Ensure Metal GPU is unlocked for TurboQuant on 16GB Macs.

    Fully automatic — no manual steps, no terminal commands, no user action
    beyond clicking "OK" on the macOS password dialog (once, ever).

    Strategy:
    1. Already unlocked → done (instant)
    2. LaunchDaemon installed but sysctl reset (reboot) → set it now via sudo -n
    3. Nothing set up yet → show native macOS password dialog, install
       LaunchDaemon (persistent) + set sysctl (immediate). One click, forever.

    Returns True if GPU is ready after this call.
    """
    if platform.system().lower() != "darwin":
        return False

    # Check RAM — only 16GB Macs need this
    try:
        mem_bytes = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip())
        ram_gb = mem_bytes // (1024 ** 3)
    except Exception:
        ram_gb = 16

    if ram_gb >= 32:
        return True  # 32GB+ doesn't need sysctl
    if ram_gb < 16:
        return False  # 8GB can't fit model

    # Already unlocked?
    if metal_gpu_available(ram_gb):
        # Ensure daemon is installed for persistence even if sysctl is already set
        if not os.path.exists(_LAUNCHDAEMON_PLIST):
            _install_gpu_daemon()
        return True

    # LaunchDaemon installed but sysctl not set (just rebooted)?
    if os.path.exists(_LAUNCHDAEMON_PLIST):
        # Try non-interactive first (sudo cache might be warm)
        try:
            subprocess.run(
                ["sudo", "-n", "sysctl", "iogpu.wired_limit_mb=14336"],
                capture_output=True, timeout=5,
            )
            if metal_gpu_available(ram_gb):
                return True
        except Exception:
            pass
        # Sudo cache cold — use osascript (native macOS dialog)
        if _sudo_via_osascript("/usr/sbin/sysctl iogpu.wired_limit_mb=14336"):
            return metal_gpu_available(ram_gb)
        return False

    # First time — install daemon + set sysctl via native macOS password dialog
    return _install_gpu_daemon()


def _install_gpu_daemon() -> bool:
    """Install LaunchDaemon for persistent GPU unlock + set sysctl now.

    Shows a native macOS password dialog via osascript. User clicks OK once,
    GPU is unlocked forever (survives reboots).
    """
    import tempfile
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".plist")
        with os.fdopen(fd, "w") as f:
            f.write(_PLIST_CONTENT)

        # Single osascript call: install daemon + set sysctl + load daemon
        cmd = (
            f"cp {tmp_path} {_LAUNCHDAEMON_PLIST} && "
            f"chmod 644 {_LAUNCHDAEMON_PLIST} && "
            f"chown root:wheel {_LAUNCHDAEMON_PLIST} && "
            f"/usr/sbin/sysctl iogpu.wired_limit_mb=14336 && "
            f"launchctl load {_LAUNCHDAEMON_PLIST}"
        )
        ok = _sudo_via_osascript(cmd)
        return ok and metal_gpu_available()
    except Exception:
        return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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


def apple_silicon_bandwidth_gbps() -> float:
    """Approx peak unified-memory bandwidth (GB/s) for THIS Mac's chip.

    Decode speed on Apple Silicon is memory-bandwidth bound (each token
    re-reads the active model weights from RAM), so this — not RAM size —
    is the primary speed lever.  Parsed from the CPU brand string so the
    estimate reflects the actual chip tier.

    Figures are Apple's advertised spec values (GB/s) where published,
    otherwise the closest verified third-party measurement.  Sources:
      M1/M1 Pro/M1 Max/M1 Ultra — Apple newsroom Oct 2021 / Mar 2022
      M2/M2 Pro/M2 Max/M2 Ultra — Apple newsroom Jan/Jun 2023
      M3/M3 Pro/M3 Max/M3 Ultra — Apple newsroom Oct/Nov 2023 + Mar 2024
        (M3 Pro regressed vs M2 Pro: 150 GB/s vs 200 GB/s)
      M4/M4 Pro/M4 Max           — Apple newsroom Oct 2024
        (M4 base = 120 GB/s; M4 Pro = 273 GB/s; M4 Max = 546 GB/s)
      M5/M5 Pro/M5 Max           — Apple newsroom Oct 2025 / Mar 2026
        (M5 base = 153 GB/s; M5 Pro = 307 GB/s; M5 Max = 614 GB/s)
      M5 Ultra                   — expected ~1100 GB/s (unreleased as of Jun 2026)

    Conservative 150 GB/s fallback for non-Apple or unparseable brand strings.
    """
    if platform.system().lower() != "darwin":
        return 150.0
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().lower()
    except Exception:
        return 150.0
    if not brand:
        return 150.0

    # Per-chip lookup table keyed on (generation, tier).
    # Generation is the first "m<N>" token; tier is ultra/max/pro/base.
    # The M3 Pro is a known regression vs M2 Pro (150 vs 200 GB/s) so
    # simple base×multiplier tables silently overstate it; explicit values
    # avoid that class of bug.
    #
    # Ordering: check for the most-descriptive tier first (ultra > max > pro)
    # so "m1 ultra" doesn't match the "max" branch.
    _TABLE: dict[tuple[str, str], float] = {
        # M1 family
        ("m1", "ultra"): 800.0,   # Apple spec: 800 GB/s
        ("m1", "max"):   400.0,   # Apple spec: 400 GB/s
        ("m1", "pro"):   200.0,   # Apple spec: 200 GB/s
        ("m1", "base"):   68.0,   # Apple spec:  68 GB/s (LPDDR4X; notable outlier)
        # M2 family
        ("m2", "ultra"): 800.0,   # Apple spec: 800 GB/s
        ("m2", "max"):   400.0,   # Apple spec: 400 GB/s
        ("m2", "pro"):   200.0,   # Apple spec: 200 GB/s
        ("m2", "base"):  100.0,   # Apple spec: 100 GB/s
        # M3 family — M3 Pro regressed to 150 GB/s (vs 200 on M2 Pro)
        ("m3", "ultra"): 819.0,   # Apple spec: 819 GB/s
        ("m3", "max"):   400.0,   # Apple spec: 400 GB/s (16-core GPU config)
        ("m3", "pro"):   150.0,   # Apple spec: 150 GB/s (↓ from M2 Pro's 200)
        ("m3", "base"):  100.0,   # Apple spec: 100 GB/s
        # M4 family — significant bandwidth uplift at Pro/Max tiers
        # (No M4 Ultra was released.)
        ("m4", "max"):   546.0,   # Apple spec: 546 GB/s (40-core GPU)
        ("m4", "pro"):   273.0,   # Apple spec: 273 GB/s
        ("m4", "base"):  120.0,   # Apple spec: 120 GB/s
        # M5 family
        ("m5", "ultra"): 1100.0,  # Expected ~1100 GB/s (unreleased; 2× M5 Max)
        ("m5", "max"):   614.0,   # Apple spec: 614 GB/s
        ("m5", "pro"):   307.0,   # Apple spec: 307 GB/s
        ("m5", "base"):  153.0,   # Apple spec: 153 GB/s
    }

    gen = ""
    for g in ("m1", "m2", "m3", "m4", "m5"):
        # Match "m5" but not "m5 pro" inside the generation check — just
        # look for the token anywhere in the brand string.
        if g in brand:
            gen = g
            break  # brand strings are unique per generation

    if not gen:
        return 150.0  # non-Apple or future unrecognised generation

    if "ultra" in brand:
        tier = "ultra"
    elif "max" in brand:
        tier = "max"
    elif "pro" in brand:
        tier = "pro"
    else:
        tier = "base"

    return _TABLE.get((gen, tier), 150.0)


# ── Thermal awareness (detection + advisory only) ──────────────────
#
# The implementation lives in localcode.thermal to keep this file within
# its size budget. Re-exported here so callers can keep importing the
# thermal helpers from localcode.performance (the historical home of all
# machine-profiling / hardware-introspection code). These are advisory
# readings only; nothing here changes runtime behaviour.
from .thermal import (  # noqa: E402
    ThermalCaps,
    read_cpu_speed_limit,
    read_thermal_pressure,
    recommended_thermal_caps,
)


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
    # Apple Silicon with TurboQuant llama.cpp: use all perf cores, large batches
    if machine.system == "darwin" and machine.has_gpu:
        llama_cpp_threads = max(4, min(machine.cpu_cores, 12))
        llama_cpp_batch_size = 2048
    else:
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
        # Apple Silicon laptop with multi-region mmap fix runs the
        # validated 64K-token turbo path; size the chars budget to match
        # so config-derived num_ctx (chars/4) lines up with what the
        # runtime actually allows. Pre-2026-04-29 this was 10000 chars
        # (~2500 tokens), which left no room for system prompt + tool
        # schemas + user message and produced E3103 on the first turn.
        max_context_chars = 262144 if use_laptop_26b else (10000 if mode == "fast" else 18000)
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
        use_26b = machine.system == "darwin" and machine.has_gpu
        profile = "gemma4-26b-laptop" if use_26b else ("gemma4-e4b" if mode == "fast" else "gemma4-26b-moe")
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

    # Small tier (≤16GB): TurboQuant KV compression
    if machine.tier == "small":
        kv_cache_type_k = "q8_0"
        kv_cache_type_v = "turbo4"
        expert_offload = False
        notes.append("TurboQuant KV (q8_0-K + turbo4-V).")

    # Apple Silicon: always TurboQuant llama.cpp — it's what gives us 32K context
    if machine.system == "darwin" and machine.has_gpu and profile in ("gemma4-26b-laptop", "gemma4-26b-moe"):
        runtime_provider = "llama_cpp"
        llama_cpp_gpu_layers = 999
        laptop_runtime_mode = "turbo"
        notes.append("TurboQuant llama.cpp: full GPU offload, 32K context.")

    # When using llama.cpp: enable n-gram speculation. Lookup-cache is
    # left OFF by default — see comment below for the failure mode it
    # causes on chat-style sessions.
    if runtime_provider == "llama_cpp":
        spec_type = "ngram-mod"
        draft_max = 64
        notes.append("N-gram speculative decoding enabled.")

    # `lookup_cache` (--lookup-cache-dynamic on llama-server) DELIBERATELY
    # disabled by default as of 2026-04-26. It enables prompt-lookup
    # speculative decoding — the model speculates that the next N tokens
    # match an n-gram already in the prompt. Win on tasks where the model
    # legitimately re-emits long prompt content (code rewrites, refactors)
    # but a footgun for chat: when conversation history contains a prior
    # answer to a similar question, lookup speculates that answer wholesale,
    # acceptance rate hits ~100%, and the model emits the prior reply
    # BYTE-IDENTICAL even after the user switches models. (Diagnosed from
    # MD5 match between Gemma turn and a subsequent Qwen turn on the
    # same question; prior to this commit the "Qwen plagiarised Gemma"
    # mystery was attributed to model behaviour — actually our flag.)
    # Users who want it for code-rewrite sessions can set
    # `llama_cpp_lookup_cache = true` in config.toml explicitly.
    lookup_cache = False

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
