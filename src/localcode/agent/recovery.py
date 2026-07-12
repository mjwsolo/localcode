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
    "NOT_FOUND_LOOP_LIMIT",
    "not_found_key",
    "detect_not_found_loop",
    "not_found_nudge",
    "REJECT_REREAD_LOOP_LIMIT",
    "detect_reject_reread_loop",
    "reject_reread_nudge",
    "todo_close_verification_suffix",
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

    # NARRATION detection. Only INTENT phrases count — not every short answer
    # after a tool call (a "weather in ny?" answer legitimately runs one tool
    # then replies in ~200 chars; the old broad rule deleted such answers and
    # looped). Two detectors: _NARRATION_INTENT_RE ("let me X"/"I'll Y" anywhere)
    # and _NARRATION_PRESENT_PARTICIPLE_RE (last sentence "Updating X/Now writing
    # Z" + target; past tense "Updated X" excluded).
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
    # NOTE: these are FORWARD-ONLY imperatives. They deliberately contain NO
    # self-referential loop language ("you're going in circles", "you've
    # rewritten X N times", "thrashing", "stop planning"). A small model
    # parrots whatever we inject: injecting "you're going in circles" as a
    # user-role message makes it reply "You're right, I've been going in
    # circles" — which then sits in its own context and self-conditions the
    # loop deeper (arXiv:2509.09677). So we tell it the NEXT concrete action
    # and nothing about the failure it just had.
    if signal.mode is ChurnMode.FILE_REWRITE:
        return (
            f"SYSTEM: {signal.subject} is written — move on to the next file. "
            "If it has a specific error, make ONE targeted edit_file change to "
            "the exact line; otherwise create the next file the task needs."
        )
    if signal.mode is ChurnMode.COMMAND_FAILURE:
        return (
            f"SYSTEM: `{signal.subject}` won't succeed as-is. Read its error "
            "output, fix the root cause (missing dependency, a syntax error in a "
            "config/source file, a wrong path) with an edit, then run it once."
        )
    if signal.mode is ChurnMode.PLANNING_SPIN:
        return (
            "SYSTEM: Take the next concrete action now — create or edit the next "
            "file the task needs, then run the build to verify. Write code, not "
            "a plan."
        )
    # INVESTIGATION_SPIN
    return (
        "SYSTEM: Take a concrete action now — create the next file the task "
        "needs and write real code with write_file/edit_file. Files already on "
        "disk that you did not create are not part of your task. If you truly "
        "lack information only the user has, ask ONE focused question."
    )


def rewrite_hard_stop(path, file_write_counts) -> str | None:
    """Circuit-breaker for rewrite drift: reject once a path is rewritten past
    2x the nudge limit (the nudge only advises; logs showed 16x on one file)."""
    if not path or not isinstance(path, str):
        return None
    n = file_write_counts.get(path, 0)
    if n < CHURN_FILE_WRITE_LIMIT * 2:
        return None
    return (
        f"REJECTED — HARD STOP: you have rewritten {path} {n} times this turn "
        "and it's still not right — full rewrites are drifting. Make ONE "
        "targeted edit_file change to the broken line, or leave this file and "
        "move on. Do NOT call write_file on this path again."
    )


# ── Repeated file-not-found (typo death loop) ───────────────────────
#
# read_file's own _confident_match auto-reads a real file when the typo
# is unambiguous, which breaks MOST of these loops at the source. But
# when the guessed paths are too far off to match (or point at a dir
# that doesn't exist), the model can still bounce "file not found" 20+
# times, each time with a slightly different wrong spelling/case. The
# byte-identical breakers miss it (each path differs); the churn
# detectors miss it (read_file is read-only, so it just extends the
# investigation streak). This is the read-side analogue of the bash
# command-failure guard: key on the PARENT DIRECTORY (differing typos
# of one filename share a parent) and count not-found failures.

NOT_FOUND_LOOP_LIMIT = 3
"""How many 'file not found' read_file failures against the SAME parent
directory may occur in one turn before we pin the correct path. 3 (like
CHURN_COMMAND_FAIL_LIMIT): a genuine "read A, oops try B" correction is 1-2
misses; a 3rd not-found in the same dir means the model is spraying spelling
variants of a real filename it should just list_files for."""


def not_found_key(path: str) -> str:
    """Normalize a missing read path to the family we count not-found errors
    by: its parent directory. Different typos of the same file
    (notes.md / Notes.md / note.md) all share a parent, so keying on the parent
    collapses them into one repeat count. Returns "." for a bare filename."""
    from pathlib import PurePosixPath
    p = PurePosixPath(str(path or "").strip().replace("\\", "/"))
    return str(p.parent)


def detect_not_found_loop(not_found_counts: dict[str, int]) -> tuple[str, int] | None:
    """Return (parent_dir, count) when read_file has failed 'file not found'
    NOT_FOUND_LOOP_LIMIT+ times against the same parent directory this turn —
    the model is guessing typo/case variants of a real filename instead of
    listing the directory once.

    Pure function. Caller maintains `not_found_counts` across the turn
    (parent-dir key via `not_found_key` → count of not-found read failures).
    Returns None when nothing crosses the threshold.
    """
    worst = ""
    worst_n = 0
    for key, n in not_found_counts.items():
        if n > worst_n:
            worst, worst_n = key, n
    if worst_n >= NOT_FOUND_LOOP_LIMIT:
        return worst, worst_n
    return None


def not_found_nudge(parent_dir: str, real_names: list[str] | None = None) -> str:
    """Hard, forward-only nudge pinning the correct recovery path for a
    not-found loop: list the directory ONCE, then read an EXACT name from it.

    Like `churn_nudge_for`, this is a FORWARD-ONLY imperative with NO
    echo-able self-critical loop language ("you keep failing", "stop
    guessing in circles") that a small model would parrot back into its own
    context. When `real_names` is supplied (caller listed the dir), the exact
    names are pinned so the model can't invent another variant.
    """
    listing = ""
    if real_names:
        shown = ", ".join(sorted(real_names)[:20])
        listing = (
            f" The files that actually exist in {parent_dir} are: {shown}. "
            "Read one of those EXACT names."
        )
    return (
        f"SYSTEM: read_file cannot find that file in {parent_dir}. Do not try "
        "another spelling or capitalization. Call list_files on "
        f"{parent_dir} ONCE, then read_file with an EXACT name from that "
        "listing." + listing
    )


# ── Reject → re-read → reject spin (the "already modified" / dedup death loop) ─
#
# A mutation (edit_file/write_file/multi_edit) keeps returning REJECTED — a
# read-dedup stub, a "file was modified since read" rejection, an
# old_string-not-found rejection — and the model reacts by re-reading the file
# and retrying the SAME shape, over and over. The byte-identical breakers can
# miss it (the read paths / rejection reasons vary each round) and the churn
# detector can miss it (the interleaved reads keep looking like fresh activity).
#
# Recomputed from the TRANSCRIPT TAIL each call rather than a stored counter, so
# it survives compaction — the claude-code attachments.ts discipline: a nag's
# fire condition is derived from message history, not a per-turn integer a
# compaction pass would silently reset. Single-shot per turn at the loop layer
# (a guard boolean) so we give ONE actionable redirect, then stop nagging.

REJECT_REREAD_LOOP_LIMIT = 3
"""How many REJECTED tool results (interleaved with a read_file re-read) may
accumulate in the recent transcript before we fire the one-shot redirect. 3
matches CHURN_COMMAND_FAIL_LIMIT / NOT_FOUND_LOOP_LIMIT: 1-2 rejections are a
normal "oops, re-read, fix" correction; a 3rd with the model still re-reading
means it's circling, not converging."""


def detect_reject_reread_loop(messages: list[dict], window: int = 24) -> int | None:
    """Return the count of recent REJECTED tool results when the model is stuck
    in a reject → re-read → reject spin (count >= REJECT_REREAD_LOOP_LIMIT AND at
    least one read_file was issued in the same window — the re-read tell), else
    None.

    Pure function. Reads only the last `window` messages so the cost is bounded
    and the signal is recomputed from the transcript (survives compaction). The
    read_file requirement is what distinguishes this from a plain
    repeated-failure (handled by the failure breakers): here the model KEEPS
    re-reading between rejections instead of changing strategy.
    """
    tail = messages[-window:] if (window and window > 0) else list(messages)
    rejections = 0
    saw_read = False
    for m in tail:
        role = m.get("role")
        if role == "tool":
            if str(m.get("content", "")).startswith("REJECTED"):
                rejections += 1
        elif role == "assistant":
            for tc in (m.get("tool_calls") or []):
                name = str(((tc.get("function") or {}).get("name")) or "")
                if name == "read_file":
                    saw_read = True
    if rejections >= REJECT_REREAD_LOOP_LIMIT and saw_read:
        return rejections
    return None


def reject_reread_nudge() -> str:
    """One-shot, forward-only redirect for a reject → re-read → reject spin.

    Like the churn / not-found nudges, this carries NO echo-able self-critical
    loop language ("you keep failing", "you're going in circles") that a small
    model would parrot back into its own context and self-condition on
    (arXiv:2509.09677). It names the ONE concrete recovery path and an explicit
    escape hatch (leave the file, move on) so the model doesn't keep circling.
    """
    return (
        "SYSTEM: This edit keeps being rejected and re-reading the file is not "
        "getting past it. Do this exactly once: read a SMALL focused range "
        "around the target line, copy one SHORT exact snippet from that output, "
        "and make ONE minimal edit_file change to that snippet. If it is "
        "rejected again, leave this file as it is and move on to the next step "
        "of the task — do not read it again."
    )


# ── Todo-close verification nudge (rides on the todo_write tool result) ───────
#
# When the model marks the LAST item of a 3+ item todo list completed and NONE
# of the items was itself a verification step, append a one-line reminder to the
# tool result telling it to verify before the final summary. Cheap and general:
# it rides on the tool result the model already gets back (no extra turn), and
# it fires at the exact loop-exit moment where a small model tends to declare
# victory and stop. Mirrors claude-code's TodoWriteTool verification nudge,
# generalized for a single-agent tool: there is no verifier subagent here, so we
# point at the project's own build/typecheck/tests.

_VERIFY_MENTION_RE = _re_recovery.compile(
    r"verif|typecheck|type-check|\btest|\blint|\bbuild|compile|smoke",
    _re_recovery.IGNORECASE,
)


def todo_close_verification_suffix(todos) -> str:
    """Return a one-line "verify before you finish" reminder to append to the
    todo_write tool result — but only when the model just closed out a 3+ item
    list (all completed) and NONE of the item texts mentions a verification
    step. Returns "" when the nudge doesn't apply.

    Pure function of the todos the model just sent, so it's recomputed from the
    tool call itself — no stored state to drift out of sync with compaction.
    """
    try:
        items = [t for t in (todos or []) if isinstance(t, dict)]
    except TypeError:
        return ""
    if len(items) < 3:
        return ""
    statuses = [str(t.get("status", "") or "").strip().lower() for t in items]
    if not statuses or not all(s == "completed" for s in statuses):
        return ""
    contents = " ".join(str(t.get("content", "") or "") for t in items)
    if _VERIFY_MENTION_RE.search(contents):
        return ""
    return (
        "\n\nNOTE: you just marked a 3+ item task list fully done and none of "
        "the items was a verification step. Before you write your final "
        "summary, run the project's build/typecheck/tests (or a focused smoke "
        "check) to confirm the work actually runs. Do not claim it works "
        "without verifying."
    )
