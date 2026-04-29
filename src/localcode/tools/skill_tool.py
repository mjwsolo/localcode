"""skill — model-invoked recipe loader (Voyager / agent / minimal-agent pattern)."""
from __future__ import annotations

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "skill",
        "description": (
            "Load a named recipe's instructions into the conversation. "
            "Use when the user's request matches a skill listed under "
            "'Available skills' in your system prompt. The skill body "
            "returns as the tool result; your next turn should follow it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name, e.g. 'locate' or 'run-tests'."},
            },
            "required": ["name"],
        },
    },
}


def execute(ctx: ToolContext, args: dict) -> str:
    """Load a named skill's body and return it as tool-result text. The
    next agent turn will have the instructions in-context and should
    act on them. Mirrors agent's SkillTool inline-context execution —
    no sub-agent fork.
    """
    from ..skills import load_registry, invoke_skill
    name = (args.get("name") or "").strip()
    if not name:
        return (
            "Error: skill tool requires `name`. Check the 'Available skills' "
            "block in your system prompt for valid names."
        )
    registry = load_registry(ctx.app.repo_root)
    return invoke_skill(registry, name)
