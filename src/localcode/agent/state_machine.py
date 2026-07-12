"""Deterministic lifecycle for coding tasks.

The model proposes actions; the harness owns stage transitions.  Keeping this
table outside the prompt prevents narration such as "build succeeded" from
silently becoming completion state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "TaskStage", "TaskEvent", "Transition", "normalize_stage", "transition",
    "event_for_tool",
]


class TaskStage(str, Enum):
    DISCOVER = "discover"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    REPAIR = "repair"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class TaskEvent(str, Enum):
    CONTEXT_READY = "context_ready"
    PLAN_READY = "plan_ready"
    MUTATION_SUCCEEDED = "mutation_succeeded"
    VERIFICATION_REQUESTED = "verification_requested"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    REQUIREMENTS_SATISFIED = "requirements_satisfied"
    BLOCKED = "blocked"


_TRANSITIONS: dict[tuple[TaskStage, TaskEvent], TaskStage] = {
    (TaskStage.DISCOVER, TaskEvent.CONTEXT_READY): TaskStage.PLAN,
    (TaskStage.PLAN, TaskEvent.PLAN_READY): TaskStage.IMPLEMENT,
    (TaskStage.DISCOVER, TaskEvent.MUTATION_SUCCEEDED): TaskStage.IMPLEMENT,
    (TaskStage.PLAN, TaskEvent.MUTATION_SUCCEEDED): TaskStage.IMPLEMENT,
    (TaskStage.IMPLEMENT, TaskEvent.MUTATION_SUCCEEDED): TaskStage.IMPLEMENT,
    (TaskStage.IMPLEMENT, TaskEvent.VERIFICATION_REQUESTED): TaskStage.VERIFY,
    (TaskStage.IMPLEMENT, TaskEvent.VERIFICATION_PASSED): TaskStage.VERIFY,
    (TaskStage.IMPLEMENT, TaskEvent.VERIFICATION_FAILED): TaskStage.REPAIR,
    (TaskStage.VERIFY, TaskEvent.VERIFICATION_FAILED): TaskStage.REPAIR,
    (TaskStage.REPAIR, TaskEvent.MUTATION_SUCCEEDED): TaskStage.VERIFY,
    (TaskStage.REPAIR, TaskEvent.VERIFICATION_PASSED): TaskStage.VERIFY,
    (TaskStage.VERIFY, TaskEvent.VERIFICATION_PASSED): TaskStage.VERIFY,
    (TaskStage.VERIFY, TaskEvent.REQUIREMENTS_SATISFIED): TaskStage.COMPLETE,
}


@dataclass(frozen=True)
class Transition:
    before: TaskStage
    event: TaskEvent
    after: TaskStage
    changed: bool


def normalize_stage(value: str | TaskStage | None) -> TaskStage:
    raw = str(value or "").strip().lower()
    aliases = {
        "": TaskStage.DISCOVER,
        "planning": TaskStage.PLAN,
        "scaffolding": TaskStage.PLAN,
        "running": TaskStage.IMPLEMENT,
        "implementing": TaskStage.IMPLEMENT,
        "verified": TaskStage.VERIFY,
        "done": TaskStage.COMPLETE,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return TaskStage(raw)
    except ValueError:
        return TaskStage.DISCOVER


def transition(current: str | TaskStage | None, event: TaskEvent) -> Transition:
    before = normalize_stage(current)
    if event == TaskEvent.BLOCKED:
        after = TaskStage.BLOCKED
    else:
        after = _TRANSITIONS.get((before, event), before)
    return Transition(before=before, event=event, after=after, changed=after != before)


def event_for_tool(tool_name: str, *, succeeded: bool, verification: bool = False) -> TaskEvent | None:
    """Map grounded tool facts to a lifecycle event."""
    if verification:
        return TaskEvent.VERIFICATION_PASSED if succeeded else TaskEvent.VERIFICATION_FAILED
    if succeeded and (tool_name or "").strip() in {
        "write_file", "append_file", "edit_file", "multi_edit", "apply_patch", "bash",
    }:
        return TaskEvent.MUTATION_SUCCEEDED
    return None
