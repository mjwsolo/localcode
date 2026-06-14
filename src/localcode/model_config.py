"""Central per-model × per-Mac configuration — the single source of truth.

The per-model and per-RAM-tier numbers that govern context windows, KV-cache
sizing, generation caps, compaction tiering, diffusion limits, thermal
back-off, and the per-chip bandwidth table USED to be scattered across
runtime.py, runtime_diffusion.py, compaction.py, thermal.py, performance.py,
and models_catalog.py. They were tuned carefully (OOM / KV / context / tok-s),
so this module collects them in ONE place to edit and test.

DESIGN CONTRACT
  * Pure data + small pure helpers ONLY. This module imports NOTHING from
    runtime / loop / app — it is a leaf so any module can import it without a
    cycle (verified by tests/test_architecture.py rule 1).
  * The scattered helpers (runtime._ram_ctx_ceiling, compaction tiers, …) now
    DELEGATE here; their function names / signatures / return values are
    unchanged, so call sites and the existing tests are untouched. This module
    only relocates the numbers and the small math that derives from them.
  * Behaviour-preserving: every value here is byte-identical to what it
    replaced. Changing a number here is a deliberate retune, not a refactor.
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────
# RAM-tier context ceilings (runtime.py)
# ─────────────────────────────────────────────────────────────────────
#
# The per-RAM context-window ceiling for the TurboQuant (compressed-KV)
# path. The CAP here is the RAM/KV budget; the model's trained length is a
# separate cap applied via `_model_max_ctx`. Monotonic + gap-free across all
# real Mac sizes. Tiers are (min_ram_gb, ctx) checked high→low; the first
# tier whose min_ram_gb is satisfied wins, else DEFAULT.
#
# Only 16 GB→64K and 64 GB→128K are hardware-measured; 24/32/36 hold at 64K
# pending real-hardware OOM measurement. 96 GB+ unlocks 256K (every current
# catalog model trains to >=256K, so on a big machine RAM is the binding cap).
RAM_CTX_CEILING_TIERS: tuple[tuple[int, int], ...] = (
    (96, 262144),   # 256K — 96/128/192 GB hold a 256K KV easily
    (64, 131072),   # 128K (validated; 256K KV is tight beside a Q8 model on 64 GB)
    (48, 98304),    # 96K
)
RAM_CTX_CEILING_DEFAULT = 65536  # 16-47 GB: 64K (validated on 16 GB)


def ram_ctx_ceiling(ram_gb: int) -> int:
    """Per-RAM context ceiling for the TurboQuant (compressed-KV) path."""
    for min_ram, ctx in RAM_CTX_CEILING_TIERS:
        if ram_gb >= min_ram:
            return ctx
    return RAM_CTX_CEILING_DEFAULT


# Conservative per-RAM context ceiling for the cohere2moe (North-Mini-Code)
# model, which runs on the STOCK PR-#24260 server with UNCOMPRESSED f16 KV
# (~4x heavier per token, no -fit guard). Much tighter than the turbo ladder
# so a long unconditional-reasoning turn can't grow the KV cache until OOM.
# Monotonic; same (min_ram_gb, ctx) high→low form.
COHERE_CTX_CEILING_TIERS: tuple[tuple[int, int], ...] = (
    (96, 65536),    # 64K
    (48, 49152),    # 48K
    (32, 32768),    # 32K
)
COHERE_CTX_CEILING_DEFAULT = 16384  # <32 GB: 16K (32 GB+ is the recommended floor)


def cohere_ctx_ceiling(ram_gb: int) -> int:
    """Conservative per-RAM context ceiling for the cohere2moe model."""
    for min_ram, ctx in COHERE_CTX_CEILING_TIERS:
        if ram_gb >= min_ram:
            return ctx
    return COHERE_CTX_CEILING_DEFAULT


# Per-turn generation cap for the unconditionally-reasoning cohere model:
# min(COHERE_GEN_CAP_CEILING, max(COHERE_GEN_CAP_FLOOR, ctx // 2)). Only
# TIGHTENS an otherwise-unlimited/oversized budget.
COHERE_GEN_CAP_CEILING = 8192
COHERE_GEN_CAP_FLOOR = 2048


def cohere_generation_cap(target_num_ctx: int) -> int:
    """Bounded per-turn generation budget for the cohere model."""
    return min(COHERE_GEN_CAP_CEILING, max(COHERE_GEN_CAP_FLOOR, target_num_ctx // 2))


# Sentinel returned by ctx helpers that mean "don't clamp" (unreadable GGUF,
# unknown/compressed KV dtype). Large enough that min() never picks it.
CTX_NO_CLAMP_SENTINEL = 1_000_000

# Lift the balanced/default path to the RAM ceiling only on >=32 GB machines
# (lots of headroom regardless of KV type). Below that, keep the chars-based
# value to avoid an OOM on an unvalidated small-RAM × KV-type combination.
BALANCED_RAM_LIFT_MIN_GB = 32


# ─────────────────────────────────────────────────────────────────────
# KV-cache dtype byte sizes (runtime.py)
# ─────────────────────────────────────────────────────────────────────
#
# Bytes per KV element by quant type. A MISSING key is deliberate: an unknown
# type (e.g. "turbo4") means "don't clamp — defer to the validated ceilings",
# never "guess and risk a wrong cap".
KV_DTYPE_BYTES: dict[str, float] = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0625, "q8_1": 1.125,
    "q5_1": 0.875, "q5_0": 0.8125,
    "q4_1": 0.625, "q4_0": 0.5625,
}

# RAM-aware KV-fit ceiling reserve + rounding (runtime._kv_aware_ctx_ceiling).
# Reserve the larger of KV_FIT_RESERVE_GB or KV_FIT_RESERVE_FRACTION of RAM for
# OS + app + activations. Round ctx down to a KV_FIT_CTX_MULTIPLE multiple,
# never below KV_FIT_MIN_CTX.
KV_FIT_RESERVE_GB = 3
KV_FIT_RESERVE_FRACTION = 0.15
KV_FIT_CTX_MULTIPLE = 2048
KV_FIT_MIN_CTX = 2048


# ─────────────────────────────────────────────────────────────────────
# Diffusion limits (runtime_diffusion.py)
# ─────────────────────────────────────────────────────────────────────
#
# llama-diffusion-cli `-n` is the TOTAL token budget across blocks. The agent
# loop passes num_predict=-1; treat non-positive as DIFFUSION_DEFAULT_CANVAS.
# The canvas handed to `-n` is min(_n, DIFFUSION_MAX_CANVAS).
DIFFUSION_DEFAULT_CANVAS = 2048
DIFFUSION_MAX_CANVAS = 2048

# A prompt larger than this makes DiffusionGemma denoise to empty / <unused>
# collapse even with the eb-off retry (verified ~16K+ chars). Surface E3107
# immediately rather than burn retries.
DIFFUSION_PROMPT_CHAR_LIMIT = 16000


# ─────────────────────────────────────────────────────────────────────
# Compaction tiers (compaction.py)
# ─────────────────────────────────────────────────────────────────────
#
# Below this much system RAM we do NOT spend an LLM generation on the summary
# (small machine = mid-task stall + weak model + tight window); use the
# instant deterministic summary instead. >=32 GB spends the generation.
LLM_SUMMARY_MIN_RAM_GB = 32

# Keep-recent-tokens FLOOR + CAP, and the window→keep scaling divisor.
# keep = clamp(context_window // KEEP_RECENT_WINDOW_DIVISOR,
#              KEEP_RECENT_TOKENS_DEFAULT, KEEP_RECENT_TOKENS_MAX).
KEEP_RECENT_TOKENS_DEFAULT = 6144
KEEP_RECENT_TOKENS_MAX = 49152
KEEP_RECENT_WINDOW_DIVISOR = 5


def keep_recent_for_window(context_window: int) -> int:
    """How many recent tokens to keep verbatim, scaled to the window."""
    if not context_window or context_window <= 0:
        return KEEP_RECENT_TOKENS_DEFAULT
    scaled = context_window // KEEP_RECENT_WINDOW_DIVISOR
    return max(KEEP_RECENT_TOKENS_DEFAULT, min(KEEP_RECENT_TOKENS_MAX, scaled))


# Compaction thresholds.
CHARS_PER_TOKEN = 4
RESERVE_TOKENS_DEFAULT = 4096
COMPACT_THRESHOLD_FRACTION = 0.70


# ─────────────────────────────────────────────────────────────────────
# Per-chip memory bandwidth (performance.py)
# ─────────────────────────────────────────────────────────────────────
#
# Approx peak unified-memory bandwidth (GB/s) keyed on (generation, tier).
# Decode speed on Apple Silicon is bandwidth-bound, so this is the primary
# speed lever. Apple's advertised spec values where published, else closest
# verified third-party measurement. The M3 Pro is a known regression vs M2 Pro
# (150 vs 200 GB/s) — explicit values avoid a base×multiplier table silently
# overstating it.
BANDWIDTH_FALLBACK_GBPS = 150.0  # non-Apple / unparseable brand strings
APPLE_GENERATIONS: tuple[str, ...] = ("m1", "m2", "m3", "m4", "m5")
APPLE_SILICON_BANDWIDTH_GBPS: dict[tuple[str, str], float] = {
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


# ─────────────────────────────────────────────────────────────────────
# tok/s estimate inputs (models_catalog.estimate_decode_tok_s)
# ─────────────────────────────────────────────────────────────────────
#
# Apple Silicon realises ~65-75% of advertised peak bandwidth during decode.
DECODE_REALIZED_BW_FRACTION = 0.70
# Per-token compute floor (seconds): MoE routing/dispatch (~8 ms) is higher
# than dense (~2 ms), so dense stays more bandwidth-bound. is_moe := frac < 0.5.
DECODE_MOE_ACTIVE_FRAC_THRESHOLD = 0.5
DECODE_COMPUTE_OVERHEAD_MOE_S = 0.008
DECODE_COMPUTE_OVERHEAD_DENSE_S = 0.002


# ─────────────────────────────────────────────────────────────────────
# Thermal caps (thermal.py)
# ─────────────────────────────────────────────────────────────────────
#
# Advisory back-off per heat level. Each entry:
#   throttle, thread_scale_laptop, thread_scale_other, batch_scale,
#   cooldown_seconds, reason
# (a single thread_scale where laptop/other are equal). "nominal" = no change.
THERMAL_CAPS: dict[str, dict[str, object]] = {
    "nominal": {
        "throttle": False,
        "thread_scale_laptop": 1.0, "thread_scale_other": 1.0,
        "batch_scale": 1.0, "cooldown_seconds": 0,
        "reason": "No thermal pressure; run at full configured speed.",
    },
    "fair": {
        "throttle": True,
        "thread_scale_laptop": 0.75, "thread_scale_other": 0.85,
        "batch_scale": 0.75, "cooldown_seconds": 1,
        "reason": "Mild thermal pressure; trim parallelism slightly.",
    },
    "serious": {
        "throttle": True,
        "thread_scale_laptop": 0.5, "thread_scale_other": 0.5,
        "batch_scale": 0.5, "cooldown_seconds": 3,
        "reason": "Notable thermal pressure; halve parallelism and pause between requests.",
    },
    "critical": {
        "throttle": True,
        "thread_scale_laptop": 0.25, "thread_scale_other": 0.35,
        "batch_scale": 0.25, "cooldown_seconds": 8,
        "reason": "Critical thermal pressure; minimise load and cool down between requests.",
    },
}
