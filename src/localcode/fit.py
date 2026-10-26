"""Does a model fit this Mac? One rule for the picker, the recommendation and launch.

A Mac lets the GPU wire only part of its memory: Metal's
recommendedMaxWorkingSetSize, about 2/3 of RAM on 16-36 GB Macs and about 3/4 or
more on larger ones (84% measured on a 128 GB Mac). Everything the model server
loads has to fit inside it: the GGUF files (weights, vision projector, drafter)
and what the server allocates at run time (KV cache, recurrent state, context
checkpoints, compute buffers). The run-time part was measured per model family
with macOS `footprint` near a full 64K window; formulas from GGUF headers were
off by up to 2.3x in both directions, so measurements win.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
from functools import lru_cache

# Fallback when Metal cannot be asked, or when another Mac size is being emulated.
GPU_WORKING_SET_SMALL_MAX_GB = 36
GPU_WORKING_SET_SMALL_FRACTION = 2 / 3
GPU_WORKING_SET_LARGE_FRACTION = 0.75
# Kept free inside the working set.
FIT_MARGIN_GB = 0.5

# Run-time memory beyond the files: (flat GB, fraction of the files' size), at a
# 64K window with 4 context checkpoints. Measured 2026-10-25 on the bundled
# llama-server (dev/evals/runs/research-wins/footprint.jsonl). Keyed by filename
# prefix; the first match wins.
RUNTIME_OVERHEAD: tuple[tuple[str, float, float], ...] = (
    ("gemma-4-12b", 2.2, 0.0),
    ("gemma-4-26B-A4B", 2.9, 0.0),
    ("Qwen3.6-35B-A3B", 2.35, 0.0),
    ("Qwen3.8-27B", 0.0, 0.47),
)
RUNTIME_OVERHEAD_DEFAULT = (3.0, 0.3)


@lru_cache(maxsize=1)
def _metal_working_set_bytes() -> int | None:
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "/usr/lib/libobjc.A.dylib")
        metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        device = metal.MTLCreateSystemDefaultDevice()
        if not device:
            return None
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        selector = objc.sel_registerName(b"recommendedMaxWorkingSetSize")
        send = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p))
        return int(send(device, selector)) or None
    except Exception:  # noqa: BLE001 - unknown means "use the fallback"
        return None


@lru_cache(maxsize=1)
def _physical_ram_gb() -> int | None:
    try:
        out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip()) // (1024 ** 3)
    except Exception:  # noqa: BLE001
        return None


def gpu_budget_gb(ram_gb: float) -> float:
    """GPU memory a Mac with ``ram_gb`` may use. This Mac's own Metal value when
    ``ram_gb`` is its real size; the documented fraction otherwise (emulation,
    tests, other hardware)."""
    if not os.environ.get("LOCALCODE_RAM_GB") and _physical_ram_gb() == int(ram_gb):
        real = _metal_working_set_bytes()
        if real:
            return real / (1024 ** 3)
    fraction = GPU_WORKING_SET_SMALL_FRACTION if ram_gb <= GPU_WORKING_SET_SMALL_MAX_GB else GPU_WORKING_SET_LARGE_FRACTION
    return ram_gb * fraction


def runtime_overhead_gb(filename: str | None, files_gb: float) -> float:
    name = filename or ""
    for prefix, flat, frac in RUNTIME_OVERHEAD:
        if name.startswith(prefix):
            return flat + frac * files_gb
    flat, frac = RUNTIME_OVERHEAD_DEFAULT
    return flat + frac * files_gb


def fit(files_gb: float, ram_gb: float, filename: str | None = None) -> str:
    """"fits" | "tight" | "too_big" for files totalling ``files_gb`` (weights plus
    any projector and drafter) on a Mac with ``ram_gb``."""
    need = files_gb + runtime_overhead_gb(filename, files_gb)
    budget = gpu_budget_gb(ram_gb)
    if need <= budget - FIT_MARGIN_GB:
        return "fits"
    if need <= budget:
        return "tight"
    return "too_big"
