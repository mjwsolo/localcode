"""Regression: the trailing per-round context block (ledger + fs-state + todo)
must NOT be sent as a `system` message to Qwen-family models.

Qwen 3.x's chat template raises "System message must be at the beginning" on
any non-leading system message; llama-server surfaces that as HTTP 500 on every
round once the block is non-empty, crashing the session (0.3.42 shipped Qwen
3.8 with this latent bug — a single-tool task never populated the block, so it
was missed until real multi-turn use). Gemma keeps `system` (a user turn made
it re-greet every round).
"""

import pytest

from localcode.agent.loop import ephemeral_context_role


@pytest.mark.parametrize(
    "family, expected",
    [
        ("qwen35", "user"),        # Qwen 3.8 27B (dense hybrid)
        ("qwen35moe", "user"),     # Qwen 3.6 35B-A3B (MoE) — same template lineage
        ("QWEN35", "user"),        # case-insensitive
        ("gemma4-iswa", "system"), # Gemma keeps system (re-greets as user)
        ("cohere2_moe", "system"),
        ("", "system"),            # unknown / falsy → historical default
        (None, "system"),
    ],
)
def test_ephemeral_context_role(family, expected):
    assert ephemeral_context_role(family) == expected


def test_qwen_never_gets_a_trailing_system_message():
    """The whole point: for any Qwen family string, the role is never 'system'."""
    for fam in ("qwen35", "qwen35moe", "qwen3", "qwen-anything"):
        assert ephemeral_context_role(fam) != "system"
