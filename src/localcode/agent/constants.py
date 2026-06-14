"""Agent loop constants, kept separate from the loop logic.

Pulled out of agent/__init__.py during the T0.1 split. Every name here
was previously a module-level constant in the old agent.py — nothing
has changed semantically. Keeping them in their own module lets us:

  1. Reason about policy knobs without scrolling past 1,800 lines of
     loop logic.
  2. Re-export them from `localcode.agent` so external callers
     (tests, eval, app.py) don't notice the split.
  3. Import them from sibling modules (context.py, recovery.py,
     loop.py) without creating an import cycle through the package
     `__init__`.
"""
from __future__ import annotations


__all__ = [
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
    "PROJECT_FILES",
    "READ_UNCHANGED_STUB_PREFIX",
    "CHURN_FILE_WRITE_LIMIT",
    "CHURN_COMMAND_FAIL_LIMIT",
    "CHURN_READONLY_STREAK_LIMIT",
]


# ── Turn-level caps ─────────────────────────────────────────────────

MAX_ROUNDS = 0
"""Upper bound on round-trips the agent loop takes per turn. Each
round = one model call + any tool dispatches that follow.

`0` means NO HARD CAP — matches agent (`maxTurns` opt-in, default
unlimited) and terminal coding tools (no per-turn limit). Loop termination is
delegated to the targeted safety nets that catch REAL failure
patterns rather than counting rounds:

  - 3-in-a-row identical-call breaker  (`recent_tool_sigs`)
  - same-tool > 10 in a turn           (`tool_name_counts`)
  - file-edit > 3 same path            (`file_edit_counts`)
  - investigation-spin (≥10 read-only) (`_readonly_streak`)
  - looks-fine streak (≥3 rounds)      (`_looks_fine_streak`)
  - MAX_THINKING_SECONDS / CHARS       (per-round thinking cap)
  - empty-round nudge (no content/tools)
  - `cancel_requested` + Ctrl+C        (user-initiated)

Bumped through 20 → 50 → 0 on 2026-04-26 after observing legitimate
"redesign this section" investigations need 10+ rounds, and that
every pathological loop in our telemetry trips one of the targeted
guards inside 15 rounds. MAX_ROUNDS as a hard cap was emergency
insurance with no observed claims. Set to a positive integer to
re-enable a hard ceiling (eval / batch / unattended modes may want
this; interactive sessions don't)."""

MAX_OUTPUT_TOKENS = -1
"""Per-round generation cap. -1 = unlimited; the model stops at its
natural EOS. Capping mid-stream truncates valid tool-call JSON into
unparseable garbage, which then triggers a useless recovery round. A
prior 5.8-minute round burn was caused by an 8192 cap chopping a
long-but-valid write_file call.

The system-level `--ctx-size 32768` is the absolute ceiling: prompt
+ generation cannot exceed it, so the model can't run forever even
without a per-round cap. With prompts in the 5-8K-token range, that
leaves ~24-27K tokens of generation headroom — enough for any
single source file we'd reasonably emit in one call.

Historical note: bumped from -1 → 8192 on 2026-04-26 after a
36-minute turn died with HTTP 500 when the model emitted a 42K-token
edit_file dict-literal that filled the (then 64K) context mid-decode.
That regression is no longer reachable — ctx-size is now 32768, the
mid-stream tool-arg watchdog has been gutted to a 180K hard ceiling
only, and the model's natural EOS lands well inside the budget for
normal file writes."""


# ── Thinking-phase safety caps ──────────────────────────────────────
#
# Bound how much reasoning the model emits before we force a
# transition to content / tool_calls. Either cap trips a
# `_thinking_abort`, which surfaces a clear user message and ends
# the turn.
#
# Previously only the time cap existed, at 300 s — which a user
# watching a screen experiences as "frozen for 5 minutes." 90 s is
# still generous enough for a genuine slow-reason path on IQ2_M
# (~27 tok/s decode → ~2400 reasoning tokens) but roughly 3×
# snappier to bail out.
#
# The character cap catches the other failure mode: the model
# produces reasoning tokens FAST (not slow) but endlessly — tree
# diagrams, vocabulary lists, full SPAs drafted inline. Rule 23 in
# the system prompt bans those, but IQ2 compliance is imperfect;
# 4000 chars ≈ 45 lines ≈ the point where reasoning has clearly
# exceeded any legit planning budget.

MAX_THINKING_SECONDS = 90
MAX_THINKING_CHARS = 4000


# ── Tool-result size policy ─────────────────────────────────────────

RESULT_LIMITS: dict[str, int] = {
    "grep": 20_000,
    "bash": 30_000,
    "read_file": 50_000,
    "web_search": 10_000,
    "default": 50_000,
}
"""Per-tool truncation budgets (chars). Tools whose payload exceeds
their budget get truncated with a clear "[truncated N chars]" tail
so the judge / user can tell content was dropped. Tune per tool
because grep output and bash stdout have different density."""

MAX_AGGREGATE_PER_TURN = 100_000
"""Sum of tool-result chars across a single turn. If a turn ingests
more than this across its tool calls, later tool calls get their
results aggressively truncated or stubbed — prevents a single turn
from blowing past the context window via 10× big grep results."""


# ── Destructive-command detection ───────────────────────────────────

DESTRUCTIVE_PATTERNS: list[str] = [
    "rm -rf", "rm -r", "rmdir", "git push", "git reset --hard",
    "sudo ", "pip install", "npm install", "brew install",
    "docker rm", "kubectl delete", "DROP TABLE", "DELETE FROM",
    "python ", "python3 ", "node ", "npm run", "npm start",
]
"""Prefixes that trigger the approval flow in the bash tool. Matching
is substring-wise so `bash -c 'rm -rf foo'` still fires. NOT a
security boundary — a determined user can bypass via e.g. `\\rm`
or `eval` — but catches the common footgun cases."""


# ── Context-management policy ───────────────────────────────────────

COMPACT_KEEP_RECENT_TOOL_RESULTS = 4
"""Number of most-recent tool_result messages to preserve verbatim
before aging starts. Older tool_result payloads get replaced with a
brief summary stub."""

COMPACT_MIN_CONTENT_CHARS = 400
"""Only tool_result messages longer than this get considered for
aging. Short results (exit codes, one-line stdout) stay inline
because they're cheap and often load-bearing."""

REDACT_KEEP_RECENT_WRITES = 1
"""Number of most-recent write_file/edit_file tool_call args to
preserve verbatim. Older ones get their `content`/`new_string` args
redacted with a stub telling the model to read_file the path if it
needs the content. Prevents the ~10× context bloat that would
otherwise accumulate from repeated full-file writes."""

REDACT_MIN_CONTENT_CHARS = 400
"""Threshold for redaction — small writes (short configs, tiny
shims) stay inline because the savings aren't worth the indirection."""


# ── Project-instruction files (load order matters) ──────────────────

PROJECT_FILES: list[str] = ["LOCALCODE.md", "localcode.md", ".localcode.md"]
"""Filenames the agent looks for at repo root for project-specific
instructions. First match wins. Kept in this list so the agent
behaviour is documented in one place and other callers (eval /
tests / setup UI) can list them without duplicating the literal."""


# ── Duplicate-read stub text ────────────────────────────────────────

READ_UNCHANGED_STUB_PREFIX = (
    "[FILE UNCHANGED — a later read_file call for this same path is in "
    "history below with the current content. Scroll forward, or re-call "
    "read_file if needed.]"
)
"""Replacement text for older duplicate read_file results, so the
model sees "the content exists further down" rather than the raw
bytes twice. `_redact_duplicate_reads` inserts this prefix."""


# ── Semantic-churn thresholds ───────────────────────────────────────
#
# These catch the "thrashing without converging" pattern that the
# byte-identical-call breakers miss: the model keeps WRITING a file
# (different content each time) or keeps RE-RUNNING a failing command,
# never reading the actual error and making a targeted fix. Distinct
# from the exact-repeat guards (`recent_tool_sigs`, `success_counts`)
# which only trip on identical args. Tuned conservatively so normal
# multi-step flows (a couple of edits to one file, running a build
# command twice while iterating) do NOT trip — only sustained churn.

CHURN_FILE_WRITE_LIMIT = 3
"""How many times the SAME path may be written/edited in one turn
before we nudge "stop rewriting it, read the error and make a
targeted fix." Counts every write/edit/append to the path regardless
of whether content differs — semantic churn, not byte-identical
repeats. 3 chosen because a legit flow is typically write-once then
one corrective edit (2); a 3rd full rewrite of the same file in a
turn is the churn signal (the real incident rewrote package.json ~5×)."""

CHURN_COMMAND_FAIL_LIMIT = 3
"""How many times the SAME command (keyed by its first token, e.g.
`npm`) may FAIL in one turn before we nudge "read its error output
and fix the root cause before re-running." 3 (not 2) so running a
build/install command, fixing, and one more failure while iterating
is tolerated; the 3rd failure of the same command family means the
model is re-running without absorbing the error."""

CHURN_READONLY_STREAK_LIMIT = 6
"""Consecutive rounds of PURE read-only investigation (read_file /
grep / glob / list_files / web_*) with no mutating or server action
before we nudge "take a concrete action now." Tighter than the prior
generic streak of 10: a 10-round read-only run is already 5+ minutes
of the user staring at a frozen screen. 6 still allows reading the
handful of files needed to understand a redesign before committing,
but interrupts a genuine investigation spin sooner."""
