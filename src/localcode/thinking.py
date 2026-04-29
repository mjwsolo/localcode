from __future__ import annotations

__all__ = ["should_use_thinking"]


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
    "planning",
    "scaffolding",
}

_NO_THINKING_STAGES = {
    "running",
    "verified",
}


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
