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
  * a todo item transitioning to completed (or the list closing out).

Everything else — reads, greps, re-edits of already-edited files, failing or
repeat-passing test runs, throwaway experiments — is not progress.

Policy: after `nudge_after` consecutive no-progress rounds, ONE wrap-up nudge
(finish the deliverables with what you have). After `nudge_after` more, the
turn ends with an honest summary. Nothing fires in the first
`PLATEAU_GRACE_ROUNDS` rounds so a slow start is never mistaken for a plateau.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PlateauTracker", "PLATEAU_GRACE_ROUNDS", "PLATEAU_NUDGE_AFTER"]

# Rounds (0-indexed) during which the detector never fires.
PLATEAU_GRACE_ROUNDS = 4
# Consecutive no-progress rounds before the wrap-up nudge; the same count again
# after the nudge ends the turn (so 2x this in total).
PLATEAU_NUDGE_AFTER = 6


@dataclass
class PlateauTracker:
    nudge_after: int = PLATEAU_NUDGE_AFTER
    grace_rounds: int = PLATEAU_GRACE_ROUNDS
    no_progress_rounds: int = 0
    total_no_progress: int = 0
    nudged: bool = False
    # Outcome of the most recent build/test this turn (None = none yet).
    last_build_passed: bool | None = None
    progress_log: list[str] = field(default_factory=list)

    # ── progress classification ──────────────────────────────────────────
    def round_made_progress(
        self,
        *,
        changed_new_file: bool,
        build_outcomes: list[bool],
        todos_before: tuple[int, int],
        todos_after: tuple[int, int],
    ) -> bool:
        """Classify one round. Also advances `last_build_passed`."""
        progress = False
        if changed_new_file:
            progress = True
            self.progress_log.append("new_file")
        for passed in build_outcomes:
            if passed and not self.last_build_passed:
                # first pass this turn, or a pass after a failure
                progress = True
                self.progress_log.append("build_pass")
            self.last_build_passed = passed
        open_before, done_before = todos_before
        open_after, done_after = todos_after
        if done_after > done_before or (open_before > 0 and open_after == 0):
            progress = True
            self.progress_log.append("todo_completed")
        return progress

    def observe_round(
        self,
        *,
        round_idx: int,
        changed_new_file: bool,
        build_outcomes: list[bool],
        todos_before: tuple[int, int],
        todos_after: tuple[int, int],
    ) -> str:
        """Feed one finished tool round. Returns "" (nothing to do),
        "wrap_up" (inject the wrap-up nudge) or "exhausted" (end the turn)."""
        if self.round_made_progress(
            changed_new_file=changed_new_file,
            build_outcomes=build_outcomes,
            todos_before=todos_before,
            todos_after=todos_after,
        ):
            self.no_progress_rounds = 0
            return ""
        self.no_progress_rounds += 1
        self.total_no_progress += 1
        if round_idx < self.grace_rounds:
            return ""
        if self.no_progress_rounds < self.nudge_after:
            return ""
        if not self.nudged:
            self.nudged = True
            self.no_progress_rounds = 0
            return "wrap_up"
        return "exhausted"

    # ── text ─────────────────────────────────────────────────────────────
    def wrap_up_nudge(self) -> str:
        return (
            f"SYSTEM: No new progress for {self.nudge_after} steps — you have "
            "reached the limit of this approach. Stop experimenting. Now: "
            "(1) write every deliverable the user named that does not exist "
            "yet, using the best approach you have; (2) where a requirement "
            "cannot be met, say so explicitly in the deliverable or README "
            "instead of retrying; (3) run the project's own check ONCE; "
            "(4) finish."
        )

    def exhausted_summary(self, *, changed_files: list[str], open_todos: list[str]) -> str:
        delivered = (
            "Delivered (files changed this turn): " + ", ".join(f"`{p}`" for p in changed_files[:20])
            + ("" if len(changed_files) <= 20 else f" (+{len(changed_files) - 20} more)")
            if changed_files
            else "Delivered: no files were changed this turn."
        )
        if open_todos:
            not_done = "Not finished: " + "; ".join(open_todos[:8]) + ("" if len(open_todos) <= 8 else "; …")
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
