"""Explicit policy snapshot for the next model round (Pi-style transition)."""
from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["NextRoundPolicy"]


@dataclass(frozen=True)
class NextRoundPolicy:
    use_thinking: bool
    recovery_reason: str = ""
    recovery_attempt: int = 0
    commit_response: bool = True

    def recover_without_thinking(self, reason: str) -> "NextRoundPolicy":
        return replace(
            self,
            use_thinking=False,
            recovery_reason=reason,
            recovery_attempt=self.recovery_attempt + 1,
            commit_response=False,
        )
