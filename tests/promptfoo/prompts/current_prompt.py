"""Promptfoo prompt loader — returns our REAL SYSTEM_PROMPT for evaluation.

Why a loader
------------
We don't want to paste the system prompt into a YAML or markdown file
(drift guaranteed). We want the eval to test the exact string the
agent ships to llama-server.

This module's `prompt(context)` function gets called by promptfoo once
per test row. `context["vars"]["user_text"]` is the scenario's user
message. We return a list of OpenAI-format chat messages:
`[{"role":"system", "content": <rendered system prompt>},
  {"role":"user",   "content": <user_text>}]`

The system prompt is rendered the same way agent.run_agent_loop renders
it at runtime — same template, same conditional blocks — but without
needing a real `LocalCodeApp` instance. We inject minimal stubs for
the {cwd}, {network_status}, {project_instructions}, {skills_block},
{reasoning_rules}, {notebook_block} slots.

This lets promptfoo evaluate prompt behavior independently of the TUI
and agent loop plumbing. A/B tests are then as simple as editing
`SYSTEM_PROMPT` in agent.py and re-running `npx promptfoo eval`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))


def _render_system_prompt() -> str:
    """Render the system prompt with reasonable defaults for every
    template slot. Matches what run_agent_loop does at runtime but
    without needing a real app instance."""
    from localcode.agent import SYSTEM_PROMPT, REASONING_RULES, NOTEBOOK_RULES_TEMPLATE

    # Sensible defaults. These can be parameterised per-scenario later
    # (e.g. to test reasoning-mode variants) by passing them through
    # promptfoo's `vars:` block.
    reasoning_rules = ""  # assume fast mode by default
    notebook_block = NOTEBOOK_RULES_TEMPLATE.format(
        notebook_dir="/tmp/eval-notebook"
    )
    return SYSTEM_PROMPT.format(
        cwd=str(_ROOT),
        project_instructions="",
        network_status="Network: ONLINE — you can download files, install packages, fetch URLs.",
        skills_block="",
        reasoning_rules=reasoning_rules,
        notebook_block=notebook_block,
    )


def prompt(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Promptfoo calls this once per test row. Returns an OpenAI-format
    chat messages list — system prompt + user message."""
    user_text = (context or {}).get("vars", {}).get("user_text", "")
    return [
        {"role": "system", "content": _render_system_prompt()},
        {"role": "user",   "content": user_text},
    ]
