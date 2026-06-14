"""Stall detection + auto-nudge recovery.

Extracted from agent/__init__.py during T0.1-d. The loop used to have
~100 lines of inline stall-detection + nudge-message-composition
mixed in with the main control flow; pulling it behind three pure
functions + an enum makes the loop read as "generate → detect → nudge
if stalled → else proceed" instead of a forest of booleans.

Three recovery failure modes we handle:

  EMPTY          — round produced only internal reasoning: no user-
                   visible content, no tool calls.
  NARRATION      — model DESCRIBED what it was about to do ("I'll
                   create the file now…") but never emitted the tool
                   call. Classic IQ2/IQ3 decode-ends-early failure.
  POST_REJECTION — previous tool_call was REJECTED / errored and the
                   model silently gave up instead of reading the
                   actionable feedback and retrying.

All three get the same recovery: append a synthetic user message
("SYSTEM: Your previous round…") that instructs the model to act,
then loop. Each mode has a distinct nudge phrasing because IQ2/IQ3
quants respond better to specific guidance matching the failure
than to a generic "please continue."

Capped by MAX_EMPTY_ROUND_RETRIES per turn — a truly broken model
doesn't get to loop forever.

Pure helpers. No side effects. `run_agent_loop` wires them into its
control flow.
"""
from __future__ import annotations

from enum import Enum


__all__ = [
    "StallMode",
    "detect_stall",
    "nudge_for",
    "MAX_EMPTY_ROUND_RETRIES",
    "ChurnMode",
    "ChurnSignal",
    "detect_churn",
    "churn_nudge_for",
]


# Maximum number of auto-nudge retries per turn. After this many
# consecutive stalled rounds we end the turn cleanly with a user-
# visible message rather than spinning. 2 chosen empirically: 1
# catches typical IQ3 narration-without-action; 2 recovers from one
# bad retry after a tool rejection. 3+ means the model is
# fundamentally confused and more nudging won't help.
MAX_EMPTY_ROUND_RETRIES = 2


class StallMode(str, Enum):
    """Which recovery failure mode the latest round exhibits.

    String values are the human-readable labels shown in telemetry
    and in the user-visible "nudging it to continue…" line.
    """
    EMPTY = "only reasoning"
    NARRATION = "intent without action"
    POST_REJECTION = "no retry after a tool rejection"


def _last_tool_failed(messages: list[dict]) -> bool:
    """True if the most recent tool-role message is a rejection or
    error result. Walks from the tail until we hit either a tool
    message (inspected) or an assistant message (means no tool
    result is waiting — not a post-rejection stall).
    """
    for m in reversed(messages):
        if m.get("role") == "tool":
            tres = str(m.get("content", ""))
            if (tres.startswith("Error") or tres.startswith("REJECTED")
                    or tres.startswith("[exit code ")
                    or "[E3" in tres[:6] or "[E4" in tres[:6]
                    or "[E5" in tres[:6]):
                return True
            return False
        if m.get("role") == "assistant":
            # Hit an assistant before any tool result → no recent
            # rejection to react to.
            return False
    return False


def detect_stall(
    tool_calls: list,
    content: str,
    tools_called_prior: list,
    messages: list[dict],
    thinking_abort: bool,
) -> StallMode | None:
    """Classify whether the latest round stalled, and how.

    Returns None if the round was productive (tool call emitted, or
    content present with no prior tool use this turn, or a thinking-
    abort already handled). Otherwise returns the `StallMode` we
    should nudge for.

    Precedence when multiple apply:
      POST_REJECTION > EMPTY > NARRATION

    POST_REJECTION wins because the nudge text specifically
    references the rejection's actionable feedback — losing that
    framing to a generic "empty" nudge would drop the signal the
    model needs to recover.
    """
    if thinking_abort:
        # Thinking-cap abort already surfaces its own "I was
        # thinking too long" message; don't double-nudge.
        return None

    if tool_calls:
        # Round was productive — no stall.
        return None

    # POST_REJECTION first (most specific).
    if _last_tool_failed(messages):
        return StallMode.POST_REJECTION

    # EMPTY: no tool calls, no content.
    if not content:
        return StallMode.EMPTY

    # NARRATION. Real failures showed the model
    # emitting hundreds of chars describing the edit it was about to
    # make, then ending the round with zero tool calls.
    #
    # Important: do NOT classify every short answer after a tool call as
    # narration. A question like "weather in ny?" legitimately uses one
    # tool, then answers in ~200 chars. The old broad `<400 chars after
    # tools` rule deleted that answer, injected an auto-nudge, and made
    # the model call the same API repeatedly. Only intent phrases count
    # as narration.
    #
    # Two complementary detectors:
    #   • _NARRATION_INTENT_RE — "let me X" / "I'll Y" / "going to Z"
    #     anywhere in the content (full scan, not just tail) catches
    #     case (1) and similar variants where the commit phrase may
    #     be mid-content rather than at the end.
    #   • _NARRATION_PRESENT_PARTICIPLE_RE — last sentence starts with
    #     "Updating X / Changing Y / Now writing Z" + target. Past
    #     tense "Updated X" / "Changed Y" deliberately excluded.
    stripped = content.strip()
    if bool(tools_called_prior) and stripped:
        if _NARRATION_INTENT_RE.search(stripped):
            return StallMode.NARRATION
        # Last-sentence check for present-participle commitments.
        # Use sentence-boundary regex (terminal punctuation FOLLOWED
        # BY whitespace) so "app.js" doesn't get split on its file
        # extension dot. Picks the last non-empty piece after splitting.
        sentences = [s for s in _SENTENCE_BOUNDARY_RE.split(stripped) if s.strip()]
        last_sentence = sentences[-1].strip() if sentences else ""
        # Strip any terminal punctuation from the last sentence so the
        # ^ anchor in the participle regex hits the first real word.
        last_sentence = last_sentence.rstrip(".!? \n")
        if last_sentence and _NARRATION_PRESENT_PARTICIPLE_RE.search(
            last_sentence[:200]
        ):
            return StallMode.NARRATION

    # Content present, not clearly a stall — probably the final
    # answer of a CHAT-style turn.
    return None


import re as _re_recovery

# Sentence boundary: terminal punctuation followed by whitespace OR
# end-of-string. The whitespace lookahead is what prevents "app.js"
# from being split on its filename dot.
_SENTENCE_BOUNDARY_RE = _re_recovery.compile(r"(?<=[.!?])\s+")

# Forward-looking commitment phrase + action verb. Past-tense
# wrap-ups ("changed", "added") naturally don't match because they
# don't have the commitment-phrase prefix.
_NARRATION_INTENT_RE = _re_recovery.compile(
    r"\b(?:let me|let's start|i'?ll|i will|i'?m going to|now (?:let me|i'?ll|i will)|going to)\b"
    r"[^.]{0,120}?\b"
    # Verb stems WITHOUT trailing \b so -ing/-ed forms match too.
    r"(?:modify|chang|updat|fix|add|remov|writ|edit|run|start|"
    r"restart|install|check|verify|test|creat|build|deploy|read|"
    r"search|list|open|launch|patch|implement)",
    _re_recovery.IGNORECASE,
)

# Present-participle action verb at the START of the last sentence,
# followed by what looks like a target (file name, path, or function
# reference). Catches "Updating <target> ..." patterns that lack any
# explicit "I'll/let me" framing. We anchor
# to the start of the last sentence so a mid-content "Updating the
# state was the bug" (analytical, not commit) doesn't trip.
_NARRATION_PRESENT_PARTICIPLE_RE = _re_recovery.compile(
    # Optional "now" / "next" prefix so "Now writing X" / "Next, adding Y"
    # also match. Allows trailing comma and whitespace flexibly.
    r"^(?:(?:now|next)[,\s]+)?"
    r"(?:updating|changing|adding|removing|writing|editing|modifying|"
    r"fixing|installing|launching|opening|building|deploying|verifying|"
    r"testing|running|starting|restarting|patching|implementing|reading|"
    r"creating)\s+\S",
    _re_recovery.IGNORECASE,
)


_NUDGE_TEXT: dict[StallMode, str] = {
    StallMode.POST_REJECTION: (
        "SYSTEM: Your previous tool call was REJECTED or "
        "returned an Error. The rejection message contained "
        "specific actionable feedback (e.g. line number of "
        "a syntax error, missing required argument, the "
        "closest matching lines for a missed old_string). "
        "READ THE ERROR carefully, then RETRY with the "
        "specific fix it suggested. The user wanted this "
        "task done — do not give up after one rejection. "
        "If write_file rejected because the edit broke "
        "syntax, RE-READ the file with read_file, then use "
        "edit_file with a small targeted change instead of "
        "another full overwrite."
    ),
    StallMode.EMPTY: (
        "SYSTEM: Your previous round produced only internal "
        "reasoning, no user-visible response and no tool call. "
        "Stop reasoning further. Emit your answer to the user "
        "NOW, or call a tool if you need more information. Do "
        "not think more — act."
    ),
    StallMode.NARRATION: (
        "SYSTEM: Your previous round described what you were "
        "ABOUT to do but did not actually do it (no tool call "
        "was emitted). Stop narrating. Emit the tool call "
        "RIGHT NOW to perform the action you just described. "
        "Do NOT repeat the description — just call the tool."
    ),
}


def nudge_for(mode: StallMode) -> str:
    """Return the synthetic user message to append for this stall
    mode. Phrased as an instruction so IQ2/IQ3 quants don't confuse
    it with a new user ask; different wording per mode so the
    guidance matches what actually went wrong.
    """
    return _NUDGE_TEXT[mode]


# ── Semantic churn (thrashing without converging) ───────────────────
#
# The stall detectors above catch rounds that produced NO action. Churn
# is the opposite: the model keeps ACTING — rewriting the same file,
# re-running the same failing command — but never converges because it
# isn't reading the error and making a targeted fix. The byte-identical
# breakers miss this (different content/output each time); we key on the
# KEY ARG — (write_file, path) / (bash, first-token) — and count repeats.

from .constants import (  # noqa: E402  (kept near the churn logic it parameterizes)
    CHURN_FILE_WRITE_LIMIT,
    CHURN_COMMAND_FAIL_LIMIT,
    CHURN_READONLY_STREAK_LIMIT,
    CHURN_PLANNING_STREAK_LIMIT,
)


class ChurnMode(str, Enum):
    """Which churn pattern the turn-so-far exhibits. String values double
    as the telemetry label and a piece of the user-visible nudge."""
    FILE_REWRITE = "repeated file rewrite"
    COMMAND_FAILURE = "repeated failing command"
    INVESTIGATION_SPIN = "read-only investigation spin"
    PLANNING_SPIN = "re-planning without progress"


class ChurnSignal:
    """The chosen churn nudge plus the raw counters behind it.

    Returned by `detect_churn` so the loop can both inject the nudge
    AND emit the counters in round_end telemetry. Not a dataclass to
    keep recovery.py dependency-free; three plain attrs is enough.
    """

    __slots__ = ("mode", "subject", "count")

    def __init__(self, mode: ChurnMode, subject: str, count: int) -> None:
        self.mode = mode
        self.subject = subject  # the file path / command token / "" for spin
        self.count = count


def command_token(command: str) -> str:
    """Normalize a shell command to the family key we count by — its
    first meaningful token (e.g. `npm`, `node`, `pytest`). Strips a
    leading env-assignment prefix (`FOO=bar npm i` → `npm`) and a
    `sudo` prefix so `sudo npm i` and `npm i` count as the same
    family. Returns "" for an empty command.
    """
    parts = str(command or "").strip().split()
    for tok in parts:
        if "=" in tok and not tok.startswith("-"):
            # env assignment prefix, skip
            continue
        if tok in {"sudo", "command", "exec", "time", "nohup"}:
            continue
        return tok
    return ""


def detect_churn(
    file_write_counts: dict[str, int],
    command_fail_counts: dict[str, int],
    readonly_streak: int,
    planning_streak: int = 0,
) -> ChurnSignal | None:
    """Decide whether the turn-so-far is churning, and how.

    Pure function. Caller maintains across the turn: file_write_counts
    (path → write/edit/append calls), command_fail_counts (command-family
    token → FAILED runs), readonly_streak (consecutive pure read-only
    rounds), planning_streak (consecutive rounds that changed no new file,
    ran no build/verify, but produced thinking/narration — re-planning
    without progress; defaults to 0 so legacy callers keep prior behaviour).

    Precedence (most→least actionable, ties lose rightward):
    COMMAND_FAILURE > FILE_REWRITE > INVESTIGATION_SPIN > PLANNING_SPIN.

    Returns None when nothing crosses its threshold.
    """
    # COMMAND_FAILURE — most specific / actionable.
    worst_cmd = ""
    worst_cmd_n = 0
    for tok, n in command_fail_counts.items():
        if tok and n > worst_cmd_n:
            worst_cmd, worst_cmd_n = tok, n
    if worst_cmd_n >= CHURN_COMMAND_FAIL_LIMIT:
        return ChurnSignal(ChurnMode.COMMAND_FAILURE, worst_cmd, worst_cmd_n)

    # FILE_REWRITE.
    worst_file = ""
    worst_file_n = 0
    for path, n in file_write_counts.items():
        if path and n > worst_file_n:
            worst_file, worst_file_n = path, n
    if worst_file_n >= CHURN_FILE_WRITE_LIMIT:
        return ChurnSignal(ChurnMode.FILE_REWRITE, worst_file, worst_file_n)

    # INVESTIGATION_SPIN — pure read-only tool spin.
    if readonly_streak >= CHURN_READONLY_STREAK_LIMIT:
        return ChurnSignal(ChurnMode.INVESTIGATION_SPIN, "", readonly_streak)

    # PLANNING_SPIN — softest. Re-planning (think/narrate) across rounds
    # with no file change and no build/verify; the read-only streak misses
    # this because think-only rounds (zero tools) reset it.
    if planning_streak >= CHURN_PLANNING_STREAK_LIMIT:
        return ChurnSignal(ChurnMode.PLANNING_SPIN, "", planning_streak)

    return None


def churn_nudge_for(signal: ChurnSignal) -> str:
    """Short, actionable synthetic user message for a churn signal.

    Names the concrete subject (file path / command) + count so the model
    sees what it's thrashing on, then tells it the ONE thing to do instead.
    """
    if signal.mode is ChurnMode.FILE_REWRITE:
        return (
            f"SYSTEM: You've rewritten {signal.subject} {signal.count} times "
            "this turn. Stop rewriting it. Read the ACTUAL error output from "
            "the last failure, identify the specific line/cause, and make ONE "
            "targeted edit_file change to fix that — do not overwrite the whole "
            "file again."
        )
    if signal.mode is ChurnMode.COMMAND_FAILURE:
        return (
            f"SYSTEM: `{signal.subject}` has failed {signal.count} times this "
            "turn. Re-running it will not help. Read its error output line by "
            "line, fix the ROOT CAUSE (a missing dependency, a syntax error in "
            "a config/source file, a wrong path), and only then run it again."
        )
    if signal.mode is ChurnMode.PLANNING_SPIN:
        return (
            "SYSTEM: You've spent several rounds planning and re-deriving the "
            "same approach without changing a single file or running a build. "
            "You've planned enough. Take ONE concrete action NOW: create or "
            "edit the most important file for the next step, then build/run to "
            "verify it. Do not restate the plan — execute the first step of it."
        )
    # INVESTIGATION_SPIN
    return (
        "SYSTEM: You've spent several rounds reading and searching files "
        "without taking any concrete action. You're investigating in circles. "
        "Take a concrete action NOW: make a specific edit, run the build/tests, "
        "or — if you genuinely lack information only the user has — ask ONE "
        "focused question. Do not read or grep more files this round."
    )
