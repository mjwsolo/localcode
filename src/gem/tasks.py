"""Task tracking system — model can create and track multi-step work."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from .config import ensure_home_dirs


@dataclass
class Task:
    id: str
    title: str
    status: str = "pending"  # pending | in_progress | done | blocked
    notes: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "status": self.status, "notes": self.notes, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        return cls(**{k: d.get(k, "") for k in cls.__dataclass_fields__})


class TaskStore:
    def __init__(self) -> None:
        self.path = ensure_home_dirs() / "tasks.json"
        self.tasks: list[Task] = self._load()

    def _load(self) -> list[Task]:
        if self.path.exists():
            try:
                return [Task.from_dict(t) for t in json.loads(self.path.read_text())]
            except Exception:
                pass
        return []

    def save(self) -> None:
        self.path.write_text(json.dumps([t.to_dict() for t in self.tasks], indent=2))

    def create(self, title: str) -> Task:
        task = Task(id=f"task-{len(self.tasks)+1}", title=title, created_at=time.strftime("%Y-%m-%d %H:%M"))
        self.tasks.append(task)
        self.save()
        return task

    def update(self, task_id: str, status: str = "", notes: str = "") -> str:
        for t in self.tasks:
            if t.id == task_id:
                if status:
                    t.status = status
                if notes:
                    t.notes = notes
                self.save()
                return f"Updated {task_id}: {t.status}"
        return f"Task {task_id} not found"

    def list_all(self) -> str:
        if not self.tasks:
            return "No tasks."
        lines = []
        for t in self.tasks:
            icon = {"pending": "○", "in_progress": "◉", "done": "✓", "blocked": "✗"}.get(t.status, "?")
            lines.append(f"  {icon} [{t.id}] {t.title} ({t.status})")
        return "\n".join(lines)

    def clear_done(self) -> int:
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.status != "done"]
        self.save()
        return before - len(self.tasks)
