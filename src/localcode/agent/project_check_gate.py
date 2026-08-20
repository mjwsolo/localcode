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
        self.red = ""
        self.retries = 0

    def observe(self, outcome: CheckOutcome) -> str:
        """Record a completed check. Returns RED / UNVERIFIED / OK."""
        if outcome.is_red:
            # Red is a verdict: the checker worked. Clear any earlier no-verdict.
            self.unverified = ""
            # ...but REMEMBER it. The loop feeds the diagnostics back and asks
            # for a fix, and that nudge is bounded so an unfixable project can't
            # spin. Running out of nudges is not the project turning green: it
            # used to drop the red on the floor and let the turn complete as
            # verified with a broken build. Same rule the no-verdict path
            # already states, applied to the verdict we actually have.
            self.red = outcome.detail or outcome.label or "the project typecheck reported errors"
            return RED
        if outcome.status in ("timed_out", "failed"):
            self.unverified = outcome.detail or outcome.status
            return UNVERIFIED
        # Only a checker that actually ran to GREEN clears an earlier verdict.
        # `unavailable` reaches here too, and must NOT: "no checker available"
        # is the absence of evidence, so treating it as clearing would let a
        # project that went red in round 2 launder itself clean in round 3 the
        # moment the checker stopped being detectable (node_modules removed
        # mid-turn, a tsconfig edited into an unparseable state).
        if outcome.is_verified:
            self.unverified = ""
            self.red = ""
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
        """A check that never returned a verdict — or that returned errors
        nobody fixed — is not evidence of a working build, whatever else the
        verification registry happens to hold."""
        return bool(self.unverified or self.red)

    def result_note(self) -> str:
        """Text for the FINAL RESULT, so the reason reaches the TUI and `--json`
        (unlike print_info, which is invisible in the TUI and suppressed under
        --json)."""
        if self.unverified:
            return f"\n\nProject typecheck was not verified: {self.unverified}"
        if self.red:
            return ("\n\nThe project typecheck is still reporting errors:\n"
                    f"{self.red}")
        return ""
