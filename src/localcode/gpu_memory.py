"""How much memory the GPU may wire on this Mac, and whether a model fits in it.

On Apple Silicon the Metal working set (`recommendedMaxWorkingSetSize`) is the
binding limit for a fully offloaded model: weights, KV cache and compute
buffers must all be wired inside it. macOS sets it from the RAM size (about
2/3 of RAM on small Macs, more on large ones); `iogpu.wired_limit_mb` overrides
it when nonzero. LocalCode only READS this limit. It never raises it: wiring
more than the default needs sudo, resets at reboot, and over-wiring caused
kernel panics in long agent sessions in other runtimes.

The limit comes from, in order:
  1. `sysctl iogpu.wired_limit_mb`, when a user set it (nonzero);
  2. Metal itself: the bundled llama-server's `--list-devices` prints the GPU's
     recommended working set (MTL0 total MiB), probed once per machine and OS
     build and cached;
  3. the documented default fraction of RAM.
An emulated RAM tier (LOCALCODE_RAM_GB, or an explicit `ram_gb` that is not this
machine's) always uses the default fraction for that RAM size, so evals see what
a real Mac of that size would.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from functools import lru_cache
from pathlib import Path

GIB = 1024 ** 3

# Metal compute buffers measured on this stack at -ub 512 (server.log
# "MTL0 compute buffer size"): 489 MiB (Qwen 3.6), 505 MiB, 777 MiB (larger
# batch). Rounded up to 1 GiB so one reserve covers every catalog model.
COMPUTE_RESERVE_BYTES = 1 * GIB
# The vision projector's own compute buffer (server.log "alloc_compute_meta:
# MTL0 compute buffer size = 248.10 MiB"), added on top of the mmproj file.
VISION_COMPUTE_BYTES = 256 * 1024 ** 2


def default_fraction(ram_gb: float) -> float:
    """macOS's default GPU working-set fraction. Documented figures: a 16 GB Mac
    wires about 10.7 GB and a 32 GB Mac about 21 GB (2/3); a 64 GB Mac about
    48 GB (3/4). No source gives the exact cutoff, so sizes up to 36 GB use the
    lower, safe 2/3."""
    return 2 / 3 if ram_gb <= 36 else 3 / 4


@lru_cache(maxsize=1)
def _physical_ram_bytes() -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip())
    except Exception:  # noqa: BLE001
        return None


def _sysctl_wired_limit_bytes() -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, timeout=2).stdout
        mb = int(out.strip())
    except Exception:  # noqa: BLE001
        return None
    return mb * 1024 ** 2 if mb > 0 else None


def _cache_file() -> Path:
    home = os.environ.get("LOCALCODE_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".localcode"
    return base / "cache" / "gpu_working_set.json"


def _machine_key(memsize: int) -> str:
    return f"{memsize}:{platform.mac_ver()[0]}:{platform.version()}"


_MTL_LINE = re.compile(r"MTL\d+:.*\((\d+) MiB,")


def probe_metal_working_set(memsize: int) -> int | None:
    """Metal's recommended working set in bytes, from the bundled server's
    device list. Cached per machine and OS build; None when unavailable."""
    key = _machine_key(memsize)
    cache = _cache_file()
    try:
        data = json.loads(cache.read_text())
        if data.get("key") == key and int(data.get("bytes", 0)) > 0:
            return int(data["bytes"])
    except (OSError, ValueError, TypeError):
        pass
    try:
        from .bootstrap import _turboquant_binary_path
        binary = _turboquant_binary_path()
        if binary is None or not Path(binary).is_file():
            return None
        out = subprocess.run([str(binary), "--list-devices"], capture_output=True, text=True, timeout=15)
        m = _MTL_LINE.search(out.stdout + out.stderr)
        if not m:
            return None
        value = int(m.group(1)) * 1024 ** 2
    except Exception:  # noqa: BLE001
        return None
    if not 0 < value <= memsize:
        return None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "bytes": value}))
    except OSError:
        pass
    return value


_LIVE: dict[int, int | None] = {}


def working_set_bytes(ram_gb: float | None = None) -> int:
    """The GPU memory a fully offloaded model may use on this (or an emulated) Mac."""
    from .models_catalog import _forced_ram_gb, _system_ram_gb
    if ram_gb is None:
        ram_gb = _system_ram_gb()
    emulated = _forced_ram_gb() is not None
    memsize = None if emulated or platform.system() != "Darwin" else _physical_ram_bytes()
    if memsize and int(ram_gb) == memsize // GIB:
        if memsize not in _LIVE:  # read once per process; the picker asks per row
            _LIVE[memsize] = _sysctl_wired_limit_bytes() or probe_metal_working_set(memsize)
        if _LIVE[memsize]:
            return _LIVE[memsize]
    return int(ram_gb * GIB * default_fraction(ram_gb))


def os_reserve_bytes(ram_gb: float) -> int:
    """RAM kept for macOS and other apps (same rule as the context fit)."""
    from .model_config import KV_FIT_RESERVE_FRACTION, KV_FIT_RESERVE_GB
    total = int(ram_gb * GIB)
    return max(KV_FIT_RESERVE_GB * GIB, int(total * KV_FIT_RESERVE_FRACTION))


def need_bytes(weights: int, kv_per_token: float, kv_fixed: int, ctx: int) -> int:
    """GPU memory for loaded weight files, the KV cache at `ctx`, and compute buffers."""
    return int(weights + kv_fixed + kv_per_token * ctx + COMPUTE_RESERVE_BYTES)


def fit(weights: int, ram_gb: float, kv_per_token: float, kv_fixed: int = 0) -> str:
    """"fits": weights, a working context (MIN_WORKING_CTX) and compute buffers
    sit inside the GPU working set. "tight": the server still loads (RAM holds
    the weights and a minimum context beside the OS reserve) but with a short
    context or over the GPU limit. "too_big": the launcher refuses it."""
    from .model_config import KV_FIT_MIN_CTX, MIN_WORKING_CTX
    if need_bytes(weights, kv_per_token, kv_fixed, MIN_WORKING_CTX) <= working_set_bytes(ram_gb):
        return "fits"
    if weights + kv_fixed + kv_per_token * KV_FIT_MIN_CTX + os_reserve_bytes(ram_gb) < ram_gb * GIB:
        return "tight"
    return "too_big"
