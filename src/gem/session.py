from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
import json
from pathlib import Path
import uuid

from .config import ensure_home_dirs


Message = dict[str, str]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
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

    def load(self, session_id: str) -> SessionState:
        path = self.sessions_dir / f"{session_id}.json"
        data = json.loads(path.read_text())
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
                    )
                )
        return matches[0] if matches else None

    def append_event(self, session: SessionState, event_type: str, detail: str) -> None:
        session.events.append(
            {
                "time": utc_now(),
                "type": event_type,
                "detail": detail,
            }
        )
        session.events = session.events[-200:]
        self.save(session)
