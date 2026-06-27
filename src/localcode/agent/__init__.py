"""localcode.agent — public re-export surface.

This package contains the agent turn engine, split across several
focused modules (T0.1 refactor). Everything listed below is
re-exported here so external callers can continue to import from
`localcode.agent` without caring about the internal split.

  loop.py       — `run_agent_loop`, the main entry point
  prompts.py    — SYSTEM_PROMPT, REASONING_RULES,
                  _load_project_instructions
  constants.py  — policy knobs, safety caps, tables
  context.py    — message aging / redaction / compaction pipeline
  recovery.py   — stall detection + auto-nudge
  helpers.py    — tool-dispatch + display helpers

Background: before T0.1, agent.py was a 1792-line monolith. The split
broke it into focused modules (each ≤ ~720 LoC), with __init__.py
reduced to this re-export surface — well under the 400-LoC cap the
plan sets for god modules (see dev/eval/OPTIMIZATION_PLAN.md § T0).

Unused-import warnings in this file are expected — every imported
name is intentionally re-exported. The `# noqa: F401` comments
document this for linters that honour them; Pylance doesn't, so its
"not accessed" warnings on this module are false positives.
"""
from __future__ import annotations


# Public contract for `localcode.agent`. Anything not in this list is
# internal and may be renamed / moved / deleted without warning. Names
# starting with an underscore are included for back-compat with
# tests/test_context_pipeline_e2e.py which reaches into the context
# pipeline by name — once that test is migrated to a public helper,
# those entries can come out of `__all__` and move to underscore-only
# "internal import at your own risk."
__all__ = [
    # Loop entry
    "run_agent_loop",
    # Prompts
    "SYSTEM_PROMPT",
    "REASONING_RULES",
    # Constants (policy knobs external code may want to read)
    "MAX_ROUNDS",
    "MAX_OUTPUT_TOKENS",
    "MAX_THINKING_SECONDS",
    "MAX_THINKING_CHARS",
    "RESULT_LIMITS",
    "MAX_AGGREGATE_PER_TURN",
    "DESTRUCTIVE_PATTERNS",
    "COMPACT_KEEP_RECENT_TOOL_RESULTS",
    "COMPACT_MIN_CONTENT_CHARS",
    "REDACT_KEEP_RECENT_WRITES",
    "REDACT_MIN_CONTENT_CHARS",
    "READ_UNCHANGED_STUB_PREFIX",
    # Recovery
    "StallMode",
    "detect_stall",
    "nudge_for",
    "MAX_EMPTY_ROUND_RETRIES",
    # Context pipeline — underscore-prefixed, test-only back-compat
    "_prepare_model_messages",
    "_redact_old_write_args",
    "_redact_duplicate_reads",
    "_compact_old_tool_results",
]


# ── Constants ────────────────────────────────────────────────────────
# Policy knobs / safety caps / table data moved to agent/constants.py
# during the T0.1 split. Re-exported here so external callers (tests,
# eval, app.py) that do `from localcode.agent import MAX_THINKING_SECONDS`
# keep working unchanged.

from .constants import (  # noqa: F401 — re-exports for back-compat
    MAX_ROUNDS,
    MAX_OUTPUT_TOKENS,
    MAX_THINKING_SECONDS,
    MAX_THINKING_CHARS,
    RESULT_LIMITS,
    MAX_AGGREGATE_PER_TURN,
    DESTRUCTIVE_PATTERNS,
    COMPACT_KEEP_RECENT_TOOL_RESULTS,
    COMPACT_MIN_CONTENT_CHARS,
    REDACT_KEEP_RECENT_WRITES,
    REDACT_MIN_CONTENT_CHARS,
    PROJECT_FILES as _PROJECT_FILES,
    READ_UNCHANGED_STUB_PREFIX,
)

# ── Context-management pipeline ─────────────────────────────────────
# Moved to agent/context.py during T0.1-c. Re-exported here so
# tests/test_context_pipeline_e2e.py + other callers that do
#   `from localcode.agent import _prepare_model_messages`
# keep working unchanged.

from .context import (  # noqa: F401
    _truncate_result,
    _compact_old_tool_results,
    _redact_old_write_args,
    _redact_duplicate_reads,
    _msg_bytes,
    _prepare_model_messages,
    _estimate_tokens,
    _compact_messages,
    _summarize_args,
)


# ── Stall detection + auto-nudge recovery ──────────────────────────
# Moved to agent/recovery.py during T0.1-d. The loop calls
# `detect_stall(...)` after each round and, if the round stalled,
# appends `nudge_for(mode)` as a synthetic user message before
# looping. MAX_EMPTY_ROUND_RETRIES bounds how many consecutive
# stalls we tolerate per turn.

from .recovery import (  # noqa: F401
    StallMode,
    detect_stall,
    nudge_for,
    MAX_EMPTY_ROUND_RETRIES,
)


# ── Loop-adjacent helpers ──────────────────────────────────────────
# Moved to agent/helpers.py during T0.1-e. Re-exported here so any
# internal caller that still imports via `from localcode.agent import
# _execute_tool` (or other private helpers) keeps working unchanged.
# These are not part of the public API — they're named here only to
# preserve the back-compat surface during the refactor.

from .helpers import (  # noqa: F401
    _execute_tool,
    _first_token,
    _needs_confirmation,
    _render_markdown,
    _brief_result,
    _grounded_file_summary,
    _tool_stage_label,
)

# ── Prompt templates + project-instructions loader ────────────────────────
# Moved to agent/prompts.py during the T0.1-b split. Re-exported here so
# external callers (dev/eval/prompt_variants.py, tests/promptfoo, app.py,
# tests/test_context_pipeline_e2e.py) that do
#   `from localcode.agent import SYSTEM_PROMPT`
# keep working unchanged. See agent/prompts.py for the commented
# MINIMAL-CORE variant preserved there for visual diffing.

from .prompts import (  # noqa: F401 — re-exports for back-compat
    SYSTEM_PROMPT,
    REASONING_RULES,
    _load_project_instructions,
)


# ── Tool registry ────────────────────────────────────────────────────────
#
# Every tool lives in its own file under src/localcode/tools/. That package
# assembles the registry; we just pull in the schemas (for the LLM call)
# and the dispatcher (for _execute_tool). Plan-mode gating still lives
# here because it's cross-tool policy, not tool-specific logic.



# ── Result Management ────────────────────────────────────────────────────

# ── Context Management ───────────────────────────────────────────────────

# ── Display Helpers ──────────────────────────────────────────────────────

# ── Main agent loop ─────────────────────────────────────────────────
# Moved to agent/loop.py during T0.1-f. Re-exported here so every
# existing caller that does `from localcode.agent import run_agent_loop`
# keeps working unchanged. This is the public entry point into the
# agent turn engine.

from .loop import run_agent_loop  # noqa: F401
