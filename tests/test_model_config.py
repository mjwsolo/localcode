"""model_config is the single source of truth for per-model × per-Mac config.

These tests assert that the scattered helpers DELEGATE to model_config — i.e.
each helper's output matches the model_config table/helper it now reads from.
They do NOT re-assert the tuned values themselves (the existing suites own
those); they pin the delegation so the "edit in one place" property can't
silently break (a future edit that re-inlines a constant fails here).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import model_config as mc


# ── runtime ceilings delegate to model_config ───────────────────────

def test_ram_ctx_ceiling_delegates():
    from localcode.runtime import LocalCodeRuntimeGateway as GW
    for ram in (8, 16, 24, 32, 36, 48, 64, 96, 128, 192):
        assert GW._ram_ctx_ceiling(ram) == mc.ram_ctx_ceiling(ram)


def test_cohere_ctx_ceiling_delegates():
    from localcode.runtime import LocalCodeRuntimeGateway as GW
    for ram in (8, 16, 24, 32, 36, 48, 64, 96, 128, 192):
        assert GW._cohere_ctx_ceiling(ram) == mc.cohere_ctx_ceiling(ram)


def test_kv_dtype_bytes_is_the_central_table():
    from localcode.runtime import LocalCodeRuntimeGateway as GW
    # The class attribute IS the central dict (aliased, not copied).
    assert GW._KV_DTYPE_BYTES is mc.KV_DTYPE_BYTES
    # A known compressed type is absent → caller defers (no clamp).
    assert "turbo4" not in mc.KV_DTYPE_BYTES
    assert mc.KV_DTYPE_BYTES["f16"] == 2.0


def test_cohere_generation_cap_helper():
    # Bounded between floor and ceiling, half the context window in between.
    assert mc.cohere_generation_cap(4096) == mc.COHERE_GEN_CAP_FLOOR  # 4096//2 -> 2048 floor
    assert mc.cohere_generation_cap(100000) == mc.COHERE_GEN_CAP_CEILING  # capped at 8192
    # ctx//2 (8192) lands exactly on the ceiling.
    assert mc.cohere_generation_cap(16384) == 8192


# ── compaction tiers delegate to model_config ───────────────────────

def test_compaction_constants_delegate():
    from localcode import compaction
    assert compaction.LLM_SUMMARY_MIN_RAM_GB == mc.LLM_SUMMARY_MIN_RAM_GB
    assert compaction.KEEP_RECENT_TOKENS_DEFAULT == mc.KEEP_RECENT_TOKENS_DEFAULT
    assert compaction.KEEP_RECENT_TOKENS_MAX == mc.KEEP_RECENT_TOKENS_MAX
    assert compaction.RESERVE_TOKENS_DEFAULT == mc.RESERVE_TOKENS_DEFAULT
    assert compaction.COMPACT_THRESHOLD_FRACTION == mc.COMPACT_THRESHOLD_FRACTION
    assert compaction._CHARS_PER_TOKEN == mc.CHARS_PER_TOKEN


def test_keep_recent_for_window_delegates():
    from localcode import compaction
    for win in (0, 16384, 65536, 131072, 262144):
        assert compaction._keep_recent_for_window(win) == mc.keep_recent_for_window(win)


# ── thermal caps delegate to model_config ───────────────────────────

def test_thermal_caps_delegate():
    from localcode.thermal import recommended_thermal_caps
    for level in ("nominal", "fair", "serious", "critical"):
        spec = mc.THERMAL_CAPS[level]
        caps = recommended_thermal_caps(level)  # machine=None -> "other" scale
        assert caps.throttle is bool(spec["throttle"])
        assert caps.thread_scale == spec["thread_scale_other"]
        assert caps.batch_scale == spec["batch_scale"]
        assert caps.cooldown_seconds == spec["cooldown_seconds"]
        assert caps.reason == spec["reason"]


# ── performance bandwidth table delegates to model_config ───────────

def test_bandwidth_table_is_central():
    # The function reads model_config's table; spot-check membership + fallback.
    assert mc.APPLE_SILICON_BANDWIDTH_GBPS[("m5", "max")] == 614.0
    assert mc.APPLE_SILICON_BANDWIDTH_GBPS[("m3", "pro")] == 150.0
    assert mc.BANDWIDTH_FALLBACK_GBPS == 150.0


# ── diffusion limits live in model_config ───────────────────────────

def test_diffusion_limits_central():
    assert mc.DIFFUSION_PROMPT_CHAR_LIMIT == 16000
    assert mc.DIFFUSION_DEFAULT_CANVAS == 2048
    assert mc.DIFFUSION_MAX_CANVAS == 2048


# ── leaf module: no imports from runtime / loop / app ───────────────

def test_model_config_is_import_leaf():
    import ast
    src = (Path(__file__).resolve().parent.parent
           / "src" / "localcode" / "model_config.py").read_text()
    tree = ast.parse(src)
    banned = {"runtime", "runtime_diffusion", "compaction", "performance",
              "thermal", "models_catalog", "app", "agent"}
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        for m in mods:
            tail = m.split(".")[-1]
            assert tail not in banned, f"model_config must not import {m}"
