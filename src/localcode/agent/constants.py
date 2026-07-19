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
    "CHURN_PLANNING_STREAK_LIMIT",
    "CROSS_ROUND_REPEAT_LIMIT",
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
natural EOS. This stays -1 deliberately: capping mid-stream truncates
valid tool-call JSON into unparseable garbage, which then triggers a
useless recovery round. A prior 5.8-minute round burn was caused by an
8192 cap chopping a long-but-valid write_file call. So a small per-round
token cap is the WRONG lever.

The runaway backstop is NOT ctx-size. An earlier version of this note
claimed `--ctx-size 32768` bounded prompt+generation so "the model can't
run forever" — that is FALSE for long-context models. LocalCode launches
Qwen (262K trained) with `--ctx-size 131072`; a 6K-token prompt then
leaves ~125K tokens of generation headroom ≈ ~28 minutes of nonstop
decode. A real 40-minute / no-output hang (thinking on, no answer emitted)
was traced to exactly this.

The correct backstop is the thinking runaway-guard (Feature.THINKING_CAPS
+ MAX_THINKING_SECONDS / MAX_THINKING_CHARS): it aborts a reasoning-only
phase at 10 min / ~20k tokens regardless of ctx-size, well before the
context ceiling, and surfaces a clear message. That is why the cap was
re-enabled 2026-07-19. Leaving MAX_OUTPUT_TOKENS at -1 keeps valid long
tool calls intact while the thinking guard handles the runaway case."""


# ── Thinking-phase safety caps ──────────────────────────────────────
#
# Bound how much reasoning the model emits before we force a
# transition to content / tool_calls. Either cap trips a
# `_thinking_abort`, which surfaces a clear user message and ends
# the turn.
#
# These are a RUNAWAY guard, not a reasoning budget. The earlier tight
# values (90 s / 4000 chars ≈ 1000 tokens) aborted legitimate long
# reasoning and were disabled 2026-04-27 (see features.THINKING_CAPS).
# They are re-enabled now with far more generous bounds after a real
# 29-minute / 94.8k-token runaway on Qwen Q8 with thinking on: the
# cap must never touch normal reasoning, only a pathological loop.
#
#   * 600 s (10 min): a genuine slow-reason path on a heavy Q8 model
#     (~50-75 tok/s) fits comfortably; a 29-minute loop does not.
#   * 80000 chars (~20k reasoning tokens): well above any legitimate
#     planning trace; the observed runaway was ~380k chars.
#
# With reasoning now streamed live in the TUI, the user also SEES a
# runaway and can `esc` — the cap is the backstop for unattended /
# headless runs where no one is watching.

MAX_THINKING_SECONDS = 600
MAX_THINKING_CHARS = 80000


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

CHURN_PLANNING_STREAK_LIMIT = 4
"""Consecutive rounds of PLANNING-WITHOUT-PROGRESS before we nudge
"you've planned enough; take a concrete action now."

A round counts toward this streak when ALL of:
  • it changed NO new file this turn (changed_files count didn't grow),
  • it ran NO build/test/verify command, and
  • it produced thinking or narration content (i.e. the model was
    reasoning/planning, not idle).

This catches the model that re-derives the SAME plan across many rounds
— lots of thinking, no concrete action — which the read-only-spin
signal misses because that streak resets on any round with zero tool
calls (a pure think-then-read-then-think alternation never accumulates
a pure read-only streak). Set to 4 (one higher than the read-only
limit's effective reach) so legitimate "read two files, think, read a
third, then edit" flows do NOT trip: as soon as a round changes a file
or runs a build, the streak resets to 0."""

CROSS_ROUND_REPEAT_LIMIT = 4
"""How many times the SAME (tool, canonical-args) call may run ACROSS the
turn before we nudge "you already have this result — stop repeating it."

The in-round breaker catches identical calls within ONE round; this catches
the cross-ROUND spin the logs show — read_file on the same path 53x over many
rounds, or a pkill->curl->read loop where each command succeeds (so the
command-FAILURE breaker never trips). Crucially this only NUDGES — it never
withholds the tool result (the 2026-04-29 read-dedup STUB starved legitimate
debug re-reads and hung a turn 17 min; we do not repeat that). A write/edit to
a path resets that path's read counts, so a legitimate read-after-edit isn't
counted. 4 tolerates a couple of genuine re-looks before flagging a true loop."""
