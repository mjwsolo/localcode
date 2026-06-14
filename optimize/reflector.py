"""Reflect on a candidate + its feedback, then PROPOSE an improved prompt.

This is the part that makes GEPA beat random search: instead of perturbing
the prompt blindly, an LLM *reads* the textual benchmark feedback (which
tasks failed and why) and edits the system prompt to address those specific
failure modes.

Two pieces:
  * ``build_reflection_prompt`` — pure string assembly of the meta-prompt.
    Unit-tested directly.
  * ``LLMReflector`` — calls a model via the project's runtime gateway to get
    the proposed variant. Network-bound, so tests mock the ``Reflector``
    protocol instead.

The reflector model is configurable. A STRONG model is strongly preferred
here (it has to reason about prompt design); the prompt being optimized can
target a small local model independently.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from .candidate import Candidate

# The reflector must return the new prompt inside this fence so we can extract
# it unambiguously from any surrounding chatter.
_FENCE_OPEN = "<NEW_SYSTEM_PROMPT>"
_FENCE_CLOSE = "</NEW_SYSTEM_PROMPT>"

# Placeholders the live runtime fills. The reflector MUST preserve these
# verbatim or the optimized prompt breaks at runtime, so we both instruct it
# and verify after.
REQUIRED_PLACEHOLDERS = ("{cwd}", "{network_status}", "{reasoning_rules}",
                         "{project_instructions}", "{skills_block}")


class Reflector(Protocol):
    """Maps (candidate, optional sibling context) -> a new prompt string."""

    def propose(self, parent: Candidate, context: str = "") -> str: ...


def build_reflection_prompt(parent: Candidate, context: str = "") -> str:
    """Assemble the meta-prompt sent to the reflector model. Pure function."""
    placeholders = " ".join(REQUIRED_PLACEHOLDERS)
    parts = [
        "You are optimizing the SYSTEM PROMPT of a local coding agent.",
        "A benchmark scored the current prompt and produced concrete feedback "
        "about which tasks failed and why. Rewrite the system prompt so it "
        "fixes those specific failures while keeping everything that works.",
        "",
        "Rules for your rewrite:",
        "- Keep it a system prompt for a coding agent; do not answer the tasks "
        "yourself.",
        f"- Preserve these template placeholders verbatim, somewhere in the "
        f"prompt: {placeholders}",
        "- Make targeted edits driven by the feedback. Do not pad with "
        "generic advice; brevity helps small models.",
        "- Output ONLY the new prompt, wrapped exactly as:",
        f"  {_FENCE_OPEN}",
        "  ...new system prompt...",
        f"  {_FENCE_CLOSE}",
        "",
        f"CURRENT SYSTEM PROMPT (score={parent.score}):",
        _FENCE_OPEN,
        parent.prompt,
        _FENCE_CLOSE,
        "",
        "BENCHMARK FEEDBACK:",
        parent.feedback or "(no feedback captured)",
    ]
    if context:
        parts += ["", "OTHER STRONG VARIANTS FOR REFERENCE:", context]
    return "\n".join(parts)


def extract_proposed_prompt(raw: str, fallback: str) -> str:
    """Pull the fenced prompt out of the model's reply.

    Falls back to ``fallback`` (the parent prompt) if the model didn't fence
    its answer or dropped a required placeholder — better to re-evaluate the
    parent than to inject a broken prompt into the live runtime path.
    """
    m = re.search(
        re.escape(_FENCE_OPEN) + r"(.*?)" + re.escape(_FENCE_CLOSE),
        raw,
        flags=re.DOTALL,
    )
    candidate = m.group(1).strip() if m else raw.strip()
    if not candidate:
        return fallback
    if any(ph not in candidate for ph in REQUIRED_PLACEHOLDERS):
        return fallback
    return candidate


class LLMReflector:
    """Concrete reflector backed by the project's runtime gateway.

    Lazily constructs/holds a gateway pointed at a (possibly different) model
    server. Kept import-light: the runtime is only imported on first use, so
    unit tests that mock the ``Reflector`` protocol never load it.
    """

    def __init__(self, gateway, num_predict: int = 2048) -> None:
        self._gw = gateway
        self._num_predict = num_predict

    def propose(self, parent: Candidate, context: str = "") -> str:
        meta = build_reflection_prompt(parent, context)
        content: list[str] = []
        for ev in self._gw.stream_chat_events(
            [{"role": "user", "content": meta}],
            tools=None,
            num_predict=self._num_predict,
        ):
            if ev.get("type") == "content":
                content.append(ev.get("content", ""))
        return extract_proposed_prompt("".join(content), parent.prompt)


def default_system_prompt() -> str:
    """Read the live system prompt to use as the optimization seed.

    Imports from ``localcode`` (allowed direction) so the seed is always the
    real shipped prompt, not a stale copy.
    """
    src = Path(__file__).resolve().parent.parent / "src"
    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from localcode.agent.prompts import SYSTEM_PROMPT

    return SYSTEM_PROMPT
