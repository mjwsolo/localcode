from __future__ import annotations

__all__ = ["should_use_thinking", "next_task_stage_after_tool", "adaptive_reasoning_policy"]


_AUTO_REASONING_HINTS = (
    "analyze",
    "debug",
    "diagnose",
    "compare",
    "why",
    "explain",
    "review",
    "inspect",
    "investigate",
)

_THINKING_STAGES = {
    "discover", "plan", "repair",
    "planning",
    "scaffolding",
}

_NO_THINKING_STAGES = {
    "implement", "verify", "complete",
    "running",
    "verified",
}


def adaptive_reasoning_policy(activity: str, *, unexpected_failure: bool = False,
                              decision_is_reversible: bool = True) -> bool:
    """Spend reasoning tokens on decisions and surprises, not mechanics."""
    if unexpected_failure:
        return True
    activity = (activity or "").strip().lower().replace("-", "_")
    if activity in {"planning", "plan", "debugging", "repair", "diagnosis", "architecture", "new_failure"}:
        return True
    if activity in {"read", "write", "edit", "implement", "verify", "build", "test", "poll", "wait", "status"}:
        return False
    return not decision_is_reversible


def next_task_stage_after_tool(
    current_stage: str,
    tool_name: str,
    *,
    succeeded: bool,
) -> str:
    """Leave reasoning-heavy scaffolding after successful implementation work."""
    stage = (current_stage or "").strip().lower()
    tool = (tool_name or "").strip().lower()
    if not succeeded or stage not in {"planning", "scaffolding"}:
        return stage
    if tool in {
        "write_file", "edit_file", "multi_edit", "apply_patch",
        "bash", "launch_app",
    }:
        return "running"
    return stage


def should_use_thinking(
    runtime_mode: str,
    internal_thinking_mode: str,
    *,
    goal_type: str = "",
    task_stage: str = "",
    user_text: str = "",
) -> bool:
    """Decide whether a turn should use hidden thinking tokens.

    This keeps the visible runtime mode separate from the backend
    reasoning policy. Default is off; `auto` can selectively enable
    thinking for questions / debugging / analysis tasks, while `on`
    forces it and `legacy` preserves the old `*-think` behavior.
    """
    runtime_mode = (runtime_mode or "").strip()
    policy = (internal_thinking_mode or "").strip().lower()
    if policy in {"0", "false", "no", "off", "none", ""}:
        return False
    if policy in {"1", "true", "yes", "on"}:
        # `on` FORCES thinking on every round — it respects the user's explicit
        # opt-in and never second-guesses it by stage. Stage is inferred from the
        # last completed action, not what the next round must decide, so a round
        # after a read/edit during `implement` may genuinely need reasoning;
        # gating it off here would silently break the contract. Reasoning-loop
        # prevention lives in the layers that don't lie about intent: the
        # server-side thinking budget, the periodicity detector, and the
        # verified no-thinking retry. Use `auto` for stage-aware reasoning.
        return True
    if policy == "legacy":
        return runtime_mode.endswith("-think")
    if policy == "auto":
        goal = (goal_type or "").strip().lower()
        stage = (task_stage or "").strip().lower()
        if runtime_mode.endswith("-think"):
            return True
        if stage in _THINKING_STAGES:
            return True
        if stage in _NO_THINKING_STAGES:
            return False
        if goal in {"question", "edit_existing"}:
            return True
        if goal == "build_app":
            return any(
                hint in (user_text or "").lower()
                for hint in ("plan", "scaffold", "architecture", "design", "decide", "approach")
            )
        text = (user_text or "").lower()
        if goal == "general_task" and any(hint in text for hint in _AUTO_REASONING_HINTS):
            return True
        return False
    return runtime_mode.endswith("-think")
