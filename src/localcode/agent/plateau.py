"""Plateau detector — stop a model that has reached its ceiling.

Measured on HarnessBench 2026-09-08 (17 tasks, 27B model, 20 min cap): every
task that hit the cap had already reached the oracle score other frontends got
by STOPPING early. The model kept re-editing the same file and re-running the
same experiments instead of writing the remaining deliverables. None of the
existing breakers fire on that shape: each edit has different content, each
command a different output, and the calls all succeed.

"Progress" in a round is any of:
  * a write/edit to a path not previously changed this turn (a NEW file or a
    first touch of an existing one);
  * a build/typecheck/test run that PASSED after the previous one this turn
    failed, or the first pass this turn;
  * a build/test run whose FAILURE SET moved on from the previous check this
    turn: fewer failure lines, or the same count but a signature never seen
    before this turn (a genuine debugging loop — the failure changes as the
    model fixes things). A failure signature already seen this turn is NOT
    progress: that is the true loop;
  * a todo item transitioning to completed (or the list closing out).

Everything else — reads, greps, re-edits of already-edited files, identical
failing or repeat-passing test runs, throwaway experiments — is not progress.

Policy: after `nudge_after` consecutive no-progress rounds, ONE wrap-up nudge
naming the open plan items (finish the deliverables with what you have). After
`stop_after` more, the turn ends with an honest summary — unless the most
recent check this turn PASSED within the last `PLATEAU_PASS_WINDOW` rounds, in
which case the model is told ONCE to finish now and given
`PLATEAU_FINISH_ROUNDS` more rounds. Nothing fires in the first
`PLATEAU_GRACE_ROUNDS` rounds so a slow start is never mistaken for a plateau.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "PlateauTracker",
    "PLATEAU_GRACE_ROUNDS",
    "PLATEAU_NUDGE_AFTER",
    "PLATEAU_STOP_AFTER",
    "PLATEAU_PASS_WINDOW",
    "PLATEAU_FINISH_ROUNDS",
    "failure_signature",
]

# Rounds (0-indexed) during which the detector never fires.
PLATEAU_GRACE_ROUNDS = 4
# Consecutive no-progress rounds before the wrap-up nudge.
PLATEAU_NUDGE_AFTER = 6
# Consecutive no-progress rounds AFTER the nudge before the turn ends.
PLATEAU_STOP_AFTER = 8
# A check that passed within this many rounds keeps the turn alive …
PLATEAU_PASS_WINDOW = 2
# … and the "finish now" nudge grants this many more rounds.
PLATEAU_FINISH_ROUNDS = 3


# ── failure-signature extraction ─────────────────────────────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# A path with at least one directory component: keep only the basename.
_PATH_DIR_RE = re.compile(r"(?<![\w.])(?:[\w.~-]*/)+(?=[\w.-]+)")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WS_RE = re.compile(r"\s+")
_FAILURE_LINE_RE = re.compile(
    r"FAILED|AssertionError|error TS|\berror\b|Error\b|✕|●|\bFAIL\b|Traceback|panic:",
)
_NOISE_LINE_RE = re.compile(
    # summaries that mention errors without BEING a failure line (numbers
    # are already normalised to N when this runs)
    r"^\s*(?:=+.*=+|-+.*-+|(?:N|\d+) (?:passed|failed|error)s?\b.*|found (?:N|\d+) errors?.*)\s*$",
    re.IGNORECASE,
)


def normalise_check_output(text: str) -> str:
    """Strip ANSI, path directories, numbers (durations, line numbers,
    timestamps) and collapse whitespace so two runs of the same check with
    the same *meaning* compare equal."""
    text = _ANSI_RE.sub("", str(text or ""))
    text = _PATH_DIR_RE.sub("", text)
    text = _NUM_RE.sub("N", text)
    return text


def failure_signature(output: str) -> frozenset[str]:
    """The set of normalised failure lines of a build/test output.

    Recognises pytest ("FAILED …", "AssertionError", "Error"), tsc
    ("error TSNNNN"), jest ("✕", "●") and generic "FAIL"/"error" lines.
    Count-style summary lines ("3 failed, 2 passed") are ignored so the set
    reflects WHICH things fail, not how many the runner counted. A failed
    run with no recognisable failure line falls back to its last non-empty
    line so it still has a comparable identity."""
    lines: set[str] = set()
    last_nonempty = ""
    for raw in normalise_check_output(output).splitlines():
        line = _WS_RE.sub(" ", raw).strip()
        if not line:
            continue
        last_nonempty = line
        if line.startswith(("[exit code", "<UNTRUSTED_DATA", "</UNTRUSTED_DATA")):
            # loop bookkeeping / the tool-result wrapper (whose `source=`
            # attribute echoes the command) — not part of the check's output
            continue
        if _NOISE_LINE_RE.match(line):
            continue
        if _FAILURE_LINE_RE.search(line):
            lines.add(line)
    if not lines and last_nonempty:
        lines.add(last_nonempty)
    return frozenset(lines)


@dataclass
class PlateauTracker:
    nudge_after: int = PLATEAU_NUDGE_AFTER
    stop_after: int = PLATEAU_STOP_AFTER
    grace_rounds: int = PLATEAU_GRACE_ROUNDS
    pass_window: int = PLATEAU_PASS_WINDOW
    finish_rounds: int = PLATEAU_FINISH_ROUNDS
    no_progress_rounds: int = 0
    total_no_progress: int = 0
    nudged: bool = False
    finish_nudged: bool = False
    # Outcome of the most recent build/test this turn (None = none yet).
    last_build_passed: bool | None = None
    # Rounds since the most recent PASSING check (None = none this turn).
    rounds_since_pass: int | None = None
    # Failure-set tracking for the "debugging loop" progress signal.
    last_failure_sig: frozenset[str] | None = None
    seen_failure_sigs: set[frozenset[str]] = field(default_factory=set)
    progress_log: list[str] = field(default_factory=list)

    # ── progress classification ──────────────────────────────────────────
    def round_made_progress(
        self,
        *,
        changed_new_file: bool,
        build_outcomes: list[bool],
        todos_before: tuple[int, int],
        todos_after: tuple[int, int],
        build_outputs: list[str] | None = None,
    ) -> bool:
        """Classify one round. Also advances `last_build_passed` and the
        failure-signature history. `build_outputs`, when given, is parallel
        to `build_outcomes` (the raw output of each check)."""
        progress = False
        if changed_new_file:
            progress = True
            self.progress_log.append("new_file")
        outputs = list(build_outputs or [])
        for i, passed in enumerate(build_outcomes):
            if passed:
                if not self.last_build_passed:
                    # first pass this turn, or a pass after a failure
                    progress = True
                    self.progress_log.append("build_pass")
                self.rounds_since_pass = 0
                # a pass is an empty failure set: the next failure compares to it
                self.last_failure_sig = frozenset()
            else:
                out = outputs[i] if i < len(outputs) else ""
                sig = failure_signature(out)
                prev = self.last_failure_sig
                if prev is not None and len(sig) < len(prev):
                    progress = True
                    self.progress_log.append("failures_shrank")
                elif prev is not None and sig != prev and sig not in self.seen_failure_sigs:
                    progress = True
                    self.progress_log.append("new_failure_set")
                self.seen_failure_sigs.add(sig)
                self.last_failure_sig = sig
            self.last_build_passed = passed
        open_before, done_before = todos_before
        open_after, done_after = todos_after
        if done_after > done_before or (open_before > 0 and open_after == 0):
            progress = True
            self.progress_log.append("todo_completed")
        return progress

    def recent_pass(self) -> bool:
        """True if a check passed in this round or the last `pass_window`."""
        return self.rounds_since_pass is not None and self.rounds_since_pass <= self.pass_window

    def observe_round(
        self,
        *,
        round_idx: int,
        changed_new_file: bool,
        build_outcomes: list[bool],
        todos_before: tuple[int, int],
        todos_after: tuple[int, int],
        build_outputs: list[str] | None = None,
    ) -> str:
        """Feed one finished tool round. Returns "" (nothing to do),
        "wrap_up" (inject the wrap-up nudge), "finish_now" (a check just
        passed — tell the model to finish instead of stopping it) or
        "exhausted" (end the turn)."""
        # Age the last pass BEFORE classifying so a pass in this round is 0.
        if self.rounds_since_pass is not None:
            self.rounds_since_pass += 1
        if self.round_made_progress(
            changed_new_file=changed_new_file,
            build_outcomes=build_outcomes,
            build_outputs=build_outputs,
            todos_before=todos_before,
            todos_after=todos_after,
        ):
            self.no_progress_rounds = 0
            return ""
        self.no_progress_rounds += 1
        self.total_no_progress += 1
        if round_idx < self.grace_rounds:
            return ""
        if not self.nudged:
            if self.no_progress_rounds < self.nudge_after:
                return ""
            self.nudged = True
            self.no_progress_rounds = 0
            return "wrap_up"
        if self.no_progress_rounds < self.stop_after:
            return ""
        if self.recent_pass():
            # Never end a turn right after a passing check: the model may be
            # one write away from the deliverable. Say so once, grant a few
            # rounds; if the pass is still fresh at the next threshold, keep
            # waiting rather than cutting off a finishing model.
            if not self.finish_nudged:
                self.finish_nudged = True
                # exactly `finish_rounds` more no-progress rounds before the
                # next threshold (may go negative when stop_after is small)
                self.no_progress_rounds = self.stop_after - self.finish_rounds
                return "finish_now"
            return ""
        return "exhausted"

    # ── text ─────────────────────────────────────────────────────────────
    @staticmethod
    def _open_items(open_todos: list[str] | None) -> str:
        items = [str(t).strip() for t in (open_todos or []) if str(t).strip()]
        if not items:
            return ""
        shown = items[:8]
        bullets = "\n".join(f"- {t}" for t in shown)
        more = "" if len(items) <= 8 else f"\n- … (+{len(items) - 8} more)"
        return f"Open plan items:\n{bullets}{more}"

    def wrap_up_nudge(self, open_todos: list[str] | None = None) -> str:
        items = self._open_items(open_todos)
        deliverables = (
            "(1) write every open plan item below that does not exist yet, "
            "using the best approach you have"
            if items
            else "(1) write every deliverable the user named that does not exist "
            "yet, using the best approach you have"
        )
        text = (
            f"SYSTEM: No new progress for {self.nudge_after} steps — you have "
            "reached the limit of this approach. Stop experimenting. Now: "
            f"{deliverables}; (2) where a requirement "
            "cannot be met, say so explicitly in the deliverable or README "
            "instead of retrying; (3) run the project's own check ONCE; "
            "(4) finish."
        )
        if items:
            text += "\n" + items
        return text

    def finish_now_nudge(self) -> str:
        return (
            "SYSTEM: Your check passed — finish now: write any remaining "
            "deliverable and stop."
        )

    def exhausted_summary(self, *, changed_files: list[str], open_todos: list[str]) -> str:
        delivered = (
            "Delivered (files changed this turn): " + ", ".join(f"`{p}`" for p in changed_files[:20])
            + ("" if len(changed_files) <= 20 else f" (+{len(changed_files) - 20} more)")
            if changed_files
            else "Delivered: no files were changed this turn."
        )
        items = self._open_items(open_todos)
        if items:
            not_done = "Not finished. " + items
        else:
            not_done = (
                "Not finished: any remaining requirements the model did not write "
                "out — check the deliverables above against the request."
            )
        return (
            "LocalCode stopped this turn: the model made no new progress for "
            f"{self.total_no_progress} steps (kept re-editing the same files and "
            "re-running the same checks) and did not wrap up when asked. The "
            f"task is incomplete.\n\n{delivered}\n{not_done}"
        )
