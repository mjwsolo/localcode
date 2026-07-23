"""Bounded model-generation recovery decisions, separate from loop plumbing."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RecoveryDecision", "content_length_recovery"]


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str
    attempt: int
    message: str = ""


def content_length_recovery(attempts: int, cap: int = 3) -> RecoveryDecision:
    if attempts < cap:
        return RecoveryDecision(
            "retry", "max_output_tokens", attempts + 1,
            "SYSTEM: The previous response ended at the output limit. "
            "Continue directly from the exact cutoff. Do not recap, repeat "
            "completed text, or restart the answer.",
        )
    return RecoveryDecision("exhaust", "max_output_tokens", attempts)
