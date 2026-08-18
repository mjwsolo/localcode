"""todo_write — the agent's working-memory checklist.

Small quantized models lose track of what they've already done across
rounds: they re-read the same file, re-run the same `find`, and restart from
scratch. The reactive dedup guards reject those repeats but don't tell the
model where it is in the job. This tool gives the model an explicit,
persistent task list — what's done, what's in progress, what's left — that
the agent loop feeds back into context every round (see
`agent/loop.py`), so the model can progress sequentially instead of looping.

Modelled on the TodoWrite pattern: the model sends the FULL updated list
each call (it replaces the stored one), keeps exactly ONE item in_progress,
and marks items completed the moment they're done.
"""
from __future__ import annotations

from .base import ToolContext

_STATUSES = ("pending", "in_progress", "completed")

SCHEMA = {
    "type": "function",
    "function": {
        "name": "todo_write",
        "description": (
            "Record and update your task list for a multi-step job. Call this "
            "at the START of any task with 3+ steps to lay out the plan, then "
            "again every time you finish a step or discover a new one. Send the "
            "FULL list each time — it replaces the previous one. Keep exactly "
            "ONE item 'in_progress'; mark an item 'completed' the moment it's "
            "done (don't batch). This is how you remember what you've already "
            "done and what's left, so you don't repeat work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The full, updated task list (replaces the previous one).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The task in imperative form, e.g. 'Add /login endpoint'.",
                            },
                            "status": {
                                "type": "string",
                                "enum": list(_STATUSES),
                                "description": "pending | in_progress | completed.",
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional present-continuous form shown while active, e.g. 'Adding /login endpoint'.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
}


def _normalize(todos) -> list[dict]:
    """Coerce the model's list into clean {content, status, activeForm} dicts."""
    cleaned: list[dict] = []
    for item in todos:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        status = str(item.get("status", "pending") or "pending").strip().lower()
        if status not in _STATUSES:
            status = "pending"
        active = str(item.get("activeForm", "") or "").strip() or content
        cleaned.append({"content": content, "status": status, "activeForm": active})
    return cleaned


def render_todo_reminder(todos: list[dict]) -> str:
    """Render the current list as a compact reminder for the model context.

    Used both by this tool's result and by the agent loop's per-round
    injection. Returns "" for an empty list so the first round's prompt
    prefix stays stable (cache-friendly).
    """
    if not todos:
        return ""
    marks = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}
    done = sum(1 for t in todos if t.get("status") == "completed")
    total = len(todos)
    lines = [
        f"YOUR PLAN — {done}/{total} done. This is your contract: work it top to "
        f"bottom and ADVANCE it every round. Never redo a [x] item."
    ]
    for i, t in enumerate(todos, 1):
        status = t.get("status", "pending")
        mark = marks.get(status, "[ ]")
        label = t.get("content", "")
        if status == "in_progress" and t.get("activeForm"):
            label = t["activeForm"]
        lines.append(f"  {i}. {mark} {label}")
    # Name the concrete current + next step and FORCE a real action this round —
    # a weak model left with a passive list drifts back into re-reading/looping.
    cur = next((t for t in todos if t.get("status") == "in_progress"), None)
    nxt = next((t for t in todos if t.get("status") == "pending"), None)
    if cur:
        lines.append(
            f"→ DO NOW: {cur.get('content', '')}. Take a real write_file/edit_file/"
            f"bash action toward it THIS round. The moment it's finished, mark it "
            f"completed and start the next item. Do NOT re-read files you already "
            f"have or re-explore the codebase."
        )
    elif nxt:
        lines.append(
            f"→ Nothing is in_progress. Mark '{nxt.get('content', '')}' in_progress "
            f"and DO it now — don't stop to re-plan."
        )
    else:
        lines.append("→ Every item is [x] — run the whole thing once to verify, then finish.")
    return "\n".join(lines)


def execute(ctx: ToolContext, args: dict) -> str:
    todos = args.get("todos")
    if not isinstance(todos, list):
        return (
            "Error: todo_write expects a 'todos' array of "
            "{content, status} objects. Retry with the full task list."
        )
    cleaned = _normalize(todos)
    session = getattr(ctx.app, "session", None)

    # Auto-clear once everything is done — an all-completed list means the
    # job is finished, so persist it as empty rather than a wall of [x].
    all_done = bool(cleaned) and all(t["status"] == "completed" for t in cleaned)
    if session is not None:
        session.todos = [] if all_done else cleaned

    if all_done:
        return "All tasks completed. Task list cleared."
    if not cleaned:
        if session is not None:
            session.todos = []
        return "Task list cleared."

    in_progress = [t for t in cleaned if t["status"] == "in_progress"]
    note = ""
    if len(in_progress) == 0:
        note = " Nothing is in_progress — mark the next task in_progress before you start it."
    elif len(in_progress) > 1:
        note = " More than one task is in_progress — keep it to exactly one."
    _done = sum(1 for t in cleaned if t["status"] == "completed")
    # BRIEF result for the user's screen. The full "Your task list …" reminder
    # is NOT returned here — it's injected as a model-only system message each
    # round (see loop.py render_todo_reminder), so returning it here too both
    # duplicated it for the model AND leaked the internal reminder onto the
    # user's screen.
    return f"Task list updated — {_done}/{len(cleaned)} done, {len(in_progress)} in progress.{note}"


def is_concurrency_safe(args: dict) -> bool:
    # Mutates session state; never run it alongside other tools in a batch.
    return False
