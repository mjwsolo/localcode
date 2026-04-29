"""enter_plan_mode + exit_plan_mode — planning workflow tools.

These tools create and finalize a plan artifact for the current task.
The core task/turn runtime does not depend on plan_mode; it is just
UI/telemetry state plus a shared markdown plan file.
"""
from __future__ import annotations

from .base import ToolContext


ENTER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "enter_plan_mode",
        "description": (
            "Start a planning workflow for the current task. Use this when "
            "the user's request is complex (multi-file change, new feature, "
            "migration) and you want to think through the approach before "
            "coding. After writing the plan, call exit_plan_mode."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


EXIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "exit_plan_mode",
        "description": (
            "Finish the planning workflow. The plan file contents are "
            "returned to you as the tool result; use them as guidance for "
            "the next execution steps."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def execute_enter(ctx: ToolContext, args: dict) -> str:
    from ..plans import new_slug, plan_path, plan_mode_prompt, ensure_plans_dir
    app = ctx.app
    if getattr(app, "plan_mode", False):
        slug = getattr(app, "plan_slug", None)
        if slug:
            return (
                f"Planning workflow already active. Plan file: {plan_path(slug)}. "
                "Continue exploring and refining the plan."
            )
    ensure_plans_dir()
    slug = new_slug()
    app.plan_mode = True
    app.plan_slug = slug
    path = plan_path(slug)
    return (
        "Started planning workflow.\n\n"
        f"{plan_mode_prompt(slug)}\n\n"
        f"(Plan file created for this session at {path}.)"
    )


def execute_exit(ctx: ToolContext, args: dict) -> str:
    from ..plans import read_plan, plan_path
    app = ctx.app
    slug = getattr(app, "plan_slug", None)
    if not getattr(app, "plan_mode", False):
        return "No active planning workflow — proceed with the task directly."
    if slug is None:
        app.plan_mode = False
        return "Finished the planning workflow, but no plan was written. Proceed carefully."
    content = read_plan(slug)
    app.plan_mode = False
    if not content:
        return (
            f"Finished the planning workflow, but the plan file at {plan_path(slug)} is empty. "
            "Consider /plan again and actually writing the plan this time."
        )
    return (
        "Finished the planning workflow. Plan written:\n\n"
        "```markdown\n"
        f"{content.strip()}\n"
        "```\n\n"
        "Use this as guidance for execution. Continue with the task as "
        "needed."
    )
