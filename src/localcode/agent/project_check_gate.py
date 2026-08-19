"""Turn-level state for the project-typecheck completion gate.

The gate has to answer one question honestly: *did a checker actually prove this
project builds?* Three states, not two — green, red, and NEVER ANSWERED. The
third is the dangerous one: a timeout, a checker that could not execute, or a
TypeScript reference graph that was only partially covered all used to look
exactly like green.

Kept out of `loop.py` because the decision spans the whole turn (a failure in
round 3 must still block completion in round 6) and because the loop itself has
no test seam — this does.
"""
from __future__ import annotations

from ..tools.project_check import CheckOutcome

__all__ = ["ProjectCheckGate", "RED", "UNVERIFIED", "OK"]

# What the caller should do with the round.
RED = "red"                  # real diagnostics — feed them back
UNVERIFIED = "unverified"    # no verdict — retry, then block completion
OK = "ok"                    # a checker ran green, or none was applicable


class ProjectCheckGate:
    """Remembers, for the whole turn, whether the project check ever answered."""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self.unverified = ""
        self.retries = 0

    def observe(self, outcome: CheckOutcome) -> str:
        """Record a completed check. Returns RED / UNVERIFIED / OK."""
        if outcome.is_red:
            # Red is a verdict: the checker worked. Clear any earlier no-verdict.
            self.unverified = ""
            return RED
        if outcome.status in ("timed_out", "failed"):
            self.unverified = outcome.detail or outcome.status
            return UNVERIFIED
        self.unverified = ""
        return OK

    def observe_exception(self, exc: BaseException) -> str:
        """An unexpected failure is a FAILED verification, never a clean one."""
        self.unverified = (
            f"the project typecheck could not be run ({exc.__class__.__name__})")
        return UNVERIFIED

    def consume_retry(self) -> bool:
        """True if another round should be forced. Bounded, so an unfixable
        environment can't spin — but running out of retries does NOT make the
        project verified; `blocks_completion` stays True."""
        if self.retries >= self.max_retries:
            return False
        self.retries += 1
        return True

    def blocks_completion(self) -> bool:
        """A check that never returned a verdict is not evidence of a working
        build, whatever else the verification registry happens to hold."""
        return bool(self.unverified)

    def result_note(self) -> str:
        """Text for the FINAL RESULT, so the reason reaches the TUI and `--json`
        (unlike print_info, which is invisible in the TUI and suppressed under
        --json)."""
        if not self.unverified:
            return ""
        return f"\n\nProject typecheck was not verified: {self.unverified}"
