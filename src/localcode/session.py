from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .config import ensure_home_dirs


Message = dict[str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskState:
    task_id: str
    user_request: str
    goal_type: str
    task_kind: str
    task_slug: str
    goal_summary: str
    success_criteria: list[str] = field(default_factory=list)
    status: str = "in_progress"
    current_stage: str = ""
    active_port: int = 0
    completion_status: str = ""
    blocked_reason: str = ""
    turn_count: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    final_response: str = ""


@dataclass
class SessionState:
    session_id: str
    repo_root: Path
    created_at: str
    profile: str = "e4b"
    model: str = ""
    messages: list[Message] = field(default_factory=list)
    pinned_files: list[str] = field(default_factory=list)
    last_assistant_text: str = ""
    events: list[dict[str, str]] = field(default_factory=list)
    current_task: TaskState | None = None
    recent_tasks: list[TaskState] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        home = ensure_home_dirs()
        self.sessions_dir = home / "sessions"

    def create(self, repo_root: Path, profile: str = "e4b", model: str = "") -> SessionState:
        session = SessionState(
            session_id=uuid.uuid4().hex[:12],
            repo_root=repo_root,
            created_at=utc_now(),
            profile=profile,
            model=model,
        )
        self.save(session)
        return session

    def create_task(self, session: SessionState, *, user_request: str, goal_type: str,
                    task_kind: str, task_slug: str,
                    goal_summary: str, success_criteria: list[str]) -> TaskState:
        task = TaskState(
            task_id=uuid.uuid4().hex[:12],
            user_request=user_request,
            goal_type=goal_type,
            task_kind=task_kind,
            task_slug=task_slug,
            goal_summary=goal_summary,
            success_criteria=list(success_criteria),
            turn_count=1,
            current_stage="planning" if goal_type == "build_app" else "",
        )
        session.current_task = task
        session.recent_tasks.append(task)
        session.recent_tasks = session.recent_tasks[-20:]
        self.save(session)
        return task

    def continue_task(self, session: SessionState, *, user_request: str,
                      goal_type: str | None = None,
                      goal_summary: str | None = None,
                      success_criteria: list[str] | None = None) -> TaskState | None:
        """Attach a follow-up user turn to the current task.

        Coding agents should treat short corrections as continuations
        of the task that just ran, not as fresh general chat tasks.
        Keep the original task identity and only update turn-specific
        goal details.
        """
        task = session.current_task
        if task is None:
            return None
        task.user_request = f"{task.user_request}\n\nFollow-up: {user_request}".strip()
        if goal_type is not None:
            task.goal_type = goal_type
        if goal_summary is not None:
            task.goal_summary = goal_summary
        if success_criteria is not None:
            task.success_criteria = list(success_criteria)
        task.status = "in_progress"
        task.completion_status = ""
        task.blocked_reason = ""
        task.final_response = ""
        task.turn_count += 1
        task.updated_at = utc_now()
        self.save(session)
        return task

    def update_task(self, session: SessionState, *, status: str | None = None,
                    current_stage: str | None = None,
                    active_port: int | None = None,
                    completion_status: str | None = None,
                    blocked_reason: str | None = None,
                    final_response: str | None = None,
                    increment_turn: bool = False) -> None:
        task = session.current_task
        if task is None:
            return
        if increment_turn:
            task.turn_count += 1
        if status is not None:
            task.status = status
        if current_stage is not None:
            task.current_stage = current_stage
        if active_port is not None:
            task.active_port = active_port
        if completion_status is not None:
            task.completion_status = completion_status
        if blocked_reason is not None:
            task.blocked_reason = blocked_reason
        if final_response is not None:
            task.final_response = final_response
        task.updated_at = utc_now()
        self.save(session)

    def load(self, session_id: str) -> SessionState:
        path = self.sessions_dir / f"{session_id}.json"
        data = json.loads(path.read_text())
        current_task = data.get("current_task")
        recent_tasks = data.get("recent_tasks", [])
        def _load_task(payload: dict | None) -> TaskState | None:
            if not isinstance(payload, dict):
                return None
            payload = dict(payload)
            if "task_kind" not in payload:
                payload.setdefault("task_kind", payload.get("goal_type", ""))
                payload.setdefault("task_slug", payload.get("task_id", "task"))
            payload.pop("large_write_escalations", None)
            return TaskState(**payload)
        return SessionState(
            session_id=data["session_id"],
            repo_root=Path(data["repo_root"]),
            created_at=data["created_at"],
            profile=data.get("profile", "e4b"),
            model=data.get("model", ""),
            messages=list(data.get("messages", [])),
            pinned_files=list(data.get("pinned_files", [])),
            last_assistant_text=data.get("last_assistant_text", ""),
            events=list(data.get("events", [])),
            current_task=_load_task(current_task),
            recent_tasks=[task for task in (_load_task(t) for t in recent_tasks) if task is not None],
        )

    def save(self, session: SessionState) -> Path:
        path = self.sessions_dir / f"{session.session_id}.json"
        payload = {
            "session_id": session.session_id,
            "repo_root": str(session.repo_root),
            "created_at": session.created_at,
            "profile": session.profile,
            "model": session.model,
            "messages": session.messages,
            "pinned_files": session.pinned_files,
            "last_assistant_text": session.last_assistant_text,
            "events": session.events,
            "current_task": None if session.current_task is None else {
                "task_id": session.current_task.task_id,
                "user_request": session.current_task.user_request,
                "goal_type": session.current_task.goal_type,
                "task_kind": session.current_task.task_kind,
                "task_slug": session.current_task.task_slug,
                "goal_summary": session.current_task.goal_summary,
                "success_criteria": session.current_task.success_criteria,
                "status": session.current_task.status,
                "current_stage": session.current_task.current_stage,
                "active_port": session.current_task.active_port,
                "completion_status": session.current_task.completion_status,
                "blocked_reason": session.current_task.blocked_reason,
                "turn_count": session.current_task.turn_count,
                "created_at": session.current_task.created_at,
                "updated_at": session.current_task.updated_at,
                "final_response": session.current_task.final_response,
            },
            "recent_tasks": [
                {
                    "task_id": task.task_id,
                    "user_request": task.user_request,
                    "goal_type": task.goal_type,
                    "task_kind": task.task_kind,
                    "task_slug": task.task_slug,
                    "goal_summary": task.goal_summary,
                    "success_criteria": task.success_criteria,
                    "status": task.status,
                    "current_stage": task.current_stage,
                    "active_port": task.active_port,
                    "completion_status": task.completion_status,
                    "blocked_reason": task.blocked_reason,
                    "turn_count": task.turn_count,
                    "created_at": task.created_at,
                    "updated_at": task.updated_at,
                    "final_response": task.final_response,
                }
                for task in session.recent_tasks[-20:]
            ],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def list_sessions(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            data = json.loads(path.read_text())
            rows.append((data["session_id"], data["created_at"], data["repo_root"]))
        return rows

    def latest_for_repo(self, repo_root: Path) -> SessionState | None:
        matches: list[SessionState] = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            data = json.loads(path.read_text())
            if Path(data["repo_root"]).resolve() == repo_root.resolve():
                def _load_task(payload: dict | None) -> TaskState | None:
                    if not isinstance(payload, dict):
                        return None
                    payload = dict(payload)
                    if "task_kind" not in payload:
                        payload.setdefault("task_kind", payload.get("goal_type", ""))
                        payload.setdefault("task_slug", payload.get("task_id", "task"))
                    payload.pop("large_write_escalations", None)
                    return TaskState(**payload)
                matches.append(
                    SessionState(
                        session_id=data["session_id"],
                        repo_root=Path(data["repo_root"]),
                        created_at=data["created_at"],
                        profile=data.get("profile", "e4b"),
                        model=data.get("model", ""),
                        messages=list(data.get("messages", [])),
                        pinned_files=list(data.get("pinned_files", [])),
                        last_assistant_text=data.get("last_assistant_text", ""),
                        events=list(data.get("events", [])),
                        current_task=_load_task(data.get("current_task")),
                        recent_tasks=[task for task in (_load_task(t) for t in data.get("recent_tasks", [])) if task is not None],
                    )
                )
        return matches[0] if matches else None

    def append_event(self, session: SessionState, event_type: str, detail: str, **metadata: str) -> None:
        event = {
            "time": utc_now(),
            "type": event_type,
            "detail": detail,
        }
        if metadata:
            event.update({k: str(v) for k, v in metadata.items()})
        session.events.append(event)
        session.events = session.events[-200:]
        self.save(session)
