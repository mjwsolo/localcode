"""End-of-turn cleanup, persistence, and telemetry."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .goal import GoalState
    from .prompt_context import PromptBuildResult
    from ..app import LocalCodeApp


def strip_ephemeral_nudges(messages: list[dict[str, Any]], indices: list[int]) -> None:
    """Remove synthetic mid-turn nudges before history is persisted."""
    for idx in sorted(indices, reverse=True):
        if not (0 <= idx < len(messages)):
            continue
        msg = messages[idx]
        role = msg.get("role", "") if isinstance(msg, dict) else ""
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if role == "user" and isinstance(content, str) and (
            content.startswith("SYSTEM:")
            or "Continue. Run the tool" in content
            or "had no action" in content
            or "did not run" in content
        ):
            messages.pop(idx)


def status_for_exit(loop_exit_reason: str) -> tuple[str, str]:
    completed = loop_exit_reason == "model_done" or loop_exit_reason in {
        "verified_run_or_launch",
        "run_or_launch_ready",
    }
    if completed:
        return "completed", "completed"
    if loop_exit_reason == "blocked_question":
        return "blocked_user_input", "blocked"
    if loop_exit_reason in {"user_cancel", "user_cancel_mid_tool", "stream_interrupt"}:
        return "interrupted", "interrupted"
    if loop_exit_reason.startswith("completion_gate:"):
        return "stopped_early", "failed"
    if "error" in loop_exit_reason:
        return "error", "failed"
    return "incomplete", "failed"


def finalize_turn(
    *,
    app: "LocalCodeApp",
    turn_id: str,
    task_state: Any,
    goal_state: "GoalState",
    prompt_result: "PromptBuildResult",
    final_text: str,
    loop_exit_reason: str,
    final_task_stage: str,
    started_mono: float,
    time_module: Any,
    tools_called: list[str],
    round_num: int | None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    tokens_total: int = 0,
) -> tuple[str, str, str]:
    """Emit turn_end and persist task/session state."""
    blocked_reason = final_text if loop_exit_reason == "blocked_question" else ""
    completion_status, task_status = status_for_exit(loop_exit_reason)
    try:
        from ..events import emit

        emit(
            "turn_end",
            turn_id=turn_id,
            task_id=getattr(task_state, "task_id", ""),
            task_status=task_status,
            task_stage=final_task_stage,
            task_kind=getattr(task_state, "task_kind", ""),
            task_slug=getattr(task_state, "task_slug", ""),
            blocked_reason=blocked_reason,
            duration_s=round(time_module.monotonic() - started_mono, 2),
            tools_called_count=len(tools_called),
            tools_called=tools_called[:20],
            response_chars=len(final_text),
            response=final_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total or (tokens_in + tokens_out),
            rounds=(round_num + 1) if round_num is not None else None,
            loop_exit_reason=loop_exit_reason,
            completion_status=completion_status,
            injected_skills=prompt_result.selected_skills,
            injected_skill_count=len(prompt_result.selected_skills),
            injected_skill_chars=prompt_result.selected_skill_chars,
            skill_candidates=prompt_result.skill_candidates,
        )
    except Exception as exc:
        _emit_finalization_error(turn_id, "turn_end_emit", exc)
    try:
        if hasattr(app, "store") and getattr(app, "session", None) is not None:
            app.store.update_task(
                app.session,
                status=task_status,
                current_stage=final_task_stage,
                completion_status=completion_status,
                blocked_reason=blocked_reason,
                final_response=final_text,
            )
        app._last_turn_completion_status = completion_status
        app._last_turn_task_status = task_status
        app._last_turn_task_stage = final_task_stage
        app._last_turn_blocked_reason = blocked_reason
        app._last_turn_goal = goal_state.as_dict()
    except Exception as exc:
        _emit_finalization_error(turn_id, "turn_store_update", exc)
    return completion_status, task_status, blocked_reason


def _emit_finalization_error(turn_id: str, where: str, exc: BaseException) -> None:
    try:
        from ..events import emit
        emit(
            "error",
            turn_id=turn_id,
            where=where,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    except Exception:
        pass
