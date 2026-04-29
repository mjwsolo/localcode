"""Feature flag registry.

One central list of every behaviour-modifying feature in LocalCode.
Each call site that implements a feature guard consults
`is_enabled(Feature.X)` instead of reading scattered config booleans.

Three reasons this exists:

  1. **Inventory.** Before this module existed, nobody could answer
     "how many toggleable behaviours does LocalCode have?" The
     `Feature` enum below is the answer — update it when adding a
     behaviour, not a config dict somewhere.

  2. **Ops knob.** Users can disable any feature via env var or the
     per-project `.localcode/features.toml` file when debugging
     suspected regressions. `LOCALCODE_FEATURE_CONTEXT_COMPACTION=0`
     turns context compaction off for one session without code changes.

  3. **Test surface.** Eval can flip features individually and
     measure each in isolation. Was the speed win from minimal-core
     actually from the shorter prompt, or from some other feature
     that happened to land the same week? Being able to A/B single
     features answers questions monoliths can't.

## Precedence

For any `Feature`, the effective state is determined by:

  1. Env var `LOCALCODE_FEATURE_<NAME>` (0/1/true/false/on/off)
  2. Per-project `<repo>/.localcode/features.toml` `[features]` section
  3. Global `~/.localcode/features.toml` `[features]` section
  4. `FEATURE_DEFAULTS[feature]` in this module

Env wins so a one-off shell override (`LOCALCODE_FEATURE_X=0 localcode`)
always works without editing a file. Per-project beats global so a
specific repo can opt in/out without leaking that decision to other
projects.

## What belongs here vs config.toml

Features are **behaviour toggles** — "should the pressure monitor run
at all?" Not parameters — "what's the pressure threshold?" Parameters
stay in `~/.localcode/config.toml`. If you need to tune a number,
that's config; if you need to turn a whole subsystem off for A/B,
that's a feature flag.
"""
from __future__ import annotations

import os
import tomllib
from enum import Enum
from functools import lru_cache
from pathlib import Path


__all__ = [
    "Feature",
    "FEATURE_DEFAULTS",
    "is_enabled",
    "describe",
    "all_features_state",
    "reload",
]


class Feature(str, Enum):
    """Every toggleable behaviour in LocalCode.

    Adding a feature: (1) define it here with a lowercase_with_underscores
    name that matches the env var `LOCALCODE_FEATURE_<UPPER>`, (2) add
    it to `FEATURE_DEFAULTS` below with a sensible default, (3) at the
    call site, gate the behaviour with `if is_enabled(Feature.X):`.

    The string value is what appears in logs, env var names, and
    features.toml keys. Keep them stable once shipped.
    """

    # ── Context management ─────────────────────────────────────
    CONTEXT_COMPACTION = "context_compaction"
    """LLM-summary fallback when deterministic redaction isn't
    enough. See `_compact_messages` in agent/context.py. Disable
    to see how often the deterministic pipeline alone is enough."""

    WRITE_ARG_REDACTION = "write_arg_redaction"
    """Strip older write_file / edit_file content args from
    history so they don't eat context. See `_redact_old_write_args`."""

    DUPLICATE_READ_STUB = "duplicate_read_stub"
    """Replace older read_file results with a stub when a newer
    read of the same path exists. See `_redact_duplicate_reads`."""

    TOOL_RESULT_AGING = "tool_result_aging"
    """Summarize older tool_result payloads; keep the most-recent
    N verbatim. See `_compact_old_tool_results`."""

    CONTEXT_SHIFT = "context_shift"
    """Pass `--context-shift` to llama-server so overflow rotates
    oldest tokens instead of crashing the request. Default: on.
    Turning off means any turn that spills past ctx-size hard-fails."""

    # ── Decode / thinking safety ───────────────────────────────
    THINKING_CAPS = "thinking_caps"
    """Abort thinking-only phase after MAX_THINKING_SECONDS or
    MAX_THINKING_CHARS. Prevents the Qwen IQ2_M "15 minutes of
    reasoning with no output" failure mode."""

    AUTO_NUDGE_RECOVERY = "auto_nudge_recovery"
    """Append synthetic SYSTEM: nudges when the model stalls mid-
    turn (empty round / narration-without-action / post-rejection
    give-up). See agent/recovery.py."""

    # ── Runtime / server lifecycle ─────────────────────────────
    MEMORY_PRESSURE_MONITOR = "memory_pressure_monitor"
    """Watchdog process that SIGTERMs llama-server when macOS
    reports sustained memory pressure (prevents D-state jetsam).
    See memory_guard.py."""

    RECOVERY_AUTO_RUN = "recovery_auto_run"
    """On TUI preflight, auto-run localcode-unstick recovery when
    a prior server got killed by pressure. Non-reboot path."""

    # ── Agent behaviour ────────────────────────────────────────
    PLAN_MODE = "plan_mode"
    """Read-only plan-mode overlay that forbids write/edit tools
    until the user exits. See plans/."""

    SKILLS = "skills"
    """Markdown-based prompt-template registry + auto-activate
    on trigger phrases. See skills.py."""

    NOTEBOOK = "notebook"
    """Per-session scratch directory + permission bypass for
    writes inside it. See notebook.py."""

    # ── Safety layer ───────────────────────────────────────────
    SAFETY_LAYER_HARD_BLOCKS = "safety_layer_hard_blocks"
    """Regex-based hard blocks on destructive commands (rm -rf /,
    curl | bash, mkfs, etc). See permissions_v2.SafetyLayer.
    Disabling this is a safety foot-gun — only flip during eval."""

    # ── Telemetry ──────────────────────────────────────────────
    EVENTS_TELEMETRY = "events_telemetry"
    """Structured event emission to .localcode/events.jsonl. Off
    means no turn_start / tool_call / auto_nudge records get
    written — debugging gets harder but no on-disk footprint."""

    # ── Post-write diagnostics ─────────────────────────────────
    LSP_DIAGNOSTICS = "lsp_diagnostics"
    """Run ruff / pyflakes / py_compile on .py files after the
    model writes them, display findings under "Updated files:"."""


FEATURE_DEFAULTS: dict[Feature, bool] = {
    # Context management — all ON by default; these are the wins
    # that made multi-turn coding sessions survive on 16 GB Macs.
    Feature.CONTEXT_COMPACTION: True,
    Feature.WRITE_ARG_REDACTION: True,
    Feature.DUPLICATE_READ_STUB: True,
    Feature.TOOL_RESULT_AGING: True,
    Feature.CONTEXT_SHIFT: True,

    # Decode safety. THINKING_CAPS disabled by user request 2026-04-27 —
    # the per-round caps were aborting useful long-reasoning rounds and
    # causing more failures than they prevented. Re-enable only if the
    # "15-minute reasoning" failure mode reappears.
    Feature.THINKING_CAPS: False,
    Feature.AUTO_NUDGE_RECOVERY: True,

    # Runtime — on. The pressure monitor specifically is what
    # prevents the "can't kill llama-server because it's in D
    # state after OOM" recovery nightmare.
    Feature.MEMORY_PRESSURE_MONITOR: True,
    Feature.RECOVERY_AUTO_RUN: True,

    # Agent behaviour — plan mode + skills + notebook are all on.
    Feature.PLAN_MODE: True,
    Feature.SKILLS: True,
    Feature.NOTEBOOK: True,

    # Safety — always on by default.
    Feature.SAFETY_LAYER_HARD_BLOCKS: True,

    # Telemetry — on. Small file, big debugging value.
    Feature.EVENTS_TELEMETRY: True,

    # Diagnostics — on. Cheap ruff run, visible value.
    Feature.LSP_DIAGNOSTICS: True,
}


# Completeness check at import time. Guards against forgetting to add
# a new `Feature` member to `FEATURE_DEFAULTS`. An explicit assertion
# now beats a KeyError from deep inside is_enabled() later.
_MISSING = [f for f in Feature if f not in FEATURE_DEFAULTS]
assert not _MISSING, f"FEATURE_DEFAULTS missing entries for: {_MISSING}"


# ── Override resolution ─────────────────────────────────────────────


def _truthy(raw: str | bool | None) -> bool | None:
    """Return True/False for common on/off spellings, None for unknown.
    Called on both env var strings and TOML scalar values. Rejecting
    unknown spellings (instead of silently defaulting) forces the
    user to fix typos like `LOCALCODE_FEATURE_X=enable` rather than
    silently getting the default.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    if s in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    return None


def _env_override(feature: Feature) -> bool | None:
    return _truthy(os.environ.get(f"LOCALCODE_FEATURE_{feature.name}"))


def _read_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text()).get("features", {})
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_files() -> tuple[dict, dict]:
    """Read per-project + global features.toml into dicts. Cached
    because this gets called once per `is_enabled` and we don't want
    to re-parse TOML on every tool call. Call `reload()` to bust the
    cache after editing the files.
    """
    from .paths import find_project_root, global_state_dir
    project_root = find_project_root(Path.cwd())
    proj_file = (project_root / ".localcode" / "features.toml") if project_root else None
    global_file = global_state_dir() / "features.toml"
    proj = _read_toml(proj_file) if proj_file else {}
    glob = _read_toml(global_file)
    return proj, glob


def reload() -> None:
    """Invalidate the TOML cache so the next `is_enabled()` call
    re-reads from disk. Useful for tests and for the hypothetical
    `localcode features reload` subcommand."""
    _load_files.cache_clear()


def is_enabled(feature: Feature) -> bool:
    """Return the effective on/off state of a feature.

    Resolution order: env var → per-project TOML → global TOML →
    FEATURE_DEFAULTS. First non-None override wins.
    """
    env = _env_override(feature)
    if env is not None:
        return env
    proj, glob = _load_files()
    proj_val = _truthy(proj.get(feature.value))
    if proj_val is not None:
        return proj_val
    glob_val = _truthy(glob.get(feature.value))
    if glob_val is not None:
        return glob_val
    return FEATURE_DEFAULTS[feature]


def describe(feature: Feature) -> str:
    """Return the docstring attached to a Feature enum member —
    the triple-quoted string immediately after its definition.
    Useful for a `localcode features` UI to tell the user what a
    given flag controls.
    """
    doc = getattr(Feature, feature.name).__doc__ or ""
    return " ".join(doc.split())  # normalise whitespace


def all_features_state() -> list[tuple[Feature, bool, str]]:
    """Return (feature, enabled, source) for every Feature. `source`
    is one of "env", "project", "global", "default" — lets UI show
    WHY a feature is in its current state.
    """
    proj, glob = _load_files()
    rows: list[tuple[Feature, bool, str]] = []
    for f in Feature:
        env = _env_override(f)
        if env is not None:
            rows.append((f, env, "env")); continue
        pv = _truthy(proj.get(f.value))
        if pv is not None:
            rows.append((f, pv, "project")); continue
        gv = _truthy(glob.get(f.value))
        if gv is not None:
            rows.append((f, gv, "global")); continue
        rows.append((f, FEATURE_DEFAULTS[f], "default"))
    return rows
