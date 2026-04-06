"""Ghost snapshots — save and restore conversation state at any point.

When the model hangs, user hits Ctrl+C, or you want to try a different approach:
1. Snapshot is auto-saved at the interruption point
2. User can list snapshots and resume from any one
3. File system state is also captured (for undo)

Stored in ~/.gem/snapshots/ as JSON files.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ensure_home_dirs


@dataclass
class GhostSnapshot:
    """A saved conversation state."""
    id: str
    session_id: str
    repo_root: str
    created_at: float
    reason: str              # "interrupt", "branch", "checkpoint", "auto"
    messages: list[dict]     # conversation messages at this point
    pinned_files: list[str]
    file_snapshots: dict[str, str]  # path → content at snapshot time
    model: str = ""
    label: str = ""          # user-provided label


class SnapshotStore:
    """Manage ghost snapshots."""

    def __init__(self) -> None:
        self.dir = ensure_home_dirs() / "snapshots"
        self.dir.mkdir(exist_ok=True)

    def save(self, snapshot: GhostSnapshot) -> Path:
        """Save a snapshot to disk."""
        path = self.dir / f"{snapshot.id}.json"
        data = {
            "id": snapshot.id,
            "session_id": snapshot.session_id,
            "repo_root": snapshot.repo_root,
            "created_at": snapshot.created_at,
            "reason": snapshot.reason,
            "messages": snapshot.messages,
            "pinned_files": snapshot.pinned_files,
            "file_snapshots": snapshot.file_snapshots,
            "model": snapshot.model,
            "label": snapshot.label,
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    def load(self, snapshot_id: str) -> GhostSnapshot | None:
        """Load a snapshot by ID."""
        path = self.dir / f"{snapshot_id}.json"
        if not path.is_file():
            # Try prefix match
            matches = list(self.dir.glob(f"{snapshot_id}*.json"))
            if len(matches) == 1:
                path = matches[0]
            else:
                return None
        try:
            data = json.loads(path.read_text())
            return GhostSnapshot(
                id=data["id"],
                session_id=data["session_id"],
                repo_root=data["repo_root"],
                created_at=data["created_at"],
                reason=data["reason"],
                messages=data["messages"],
                pinned_files=data.get("pinned_files", []),
                file_snapshots=data.get("file_snapshots", {}),
                model=data.get("model", ""),
                label=data.get("label", ""),
            )
        except Exception:
            return None

    def list_for_session(self, session_id: str) -> list[GhostSnapshot]:
        """List all snapshots for a session, newest first."""
        results = []
        for path in sorted(self.dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                if data.get("session_id") == session_id:
                    results.append(GhostSnapshot(
                        id=data["id"],
                        session_id=data["session_id"],
                        repo_root=data["repo_root"],
                        created_at=data["created_at"],
                        reason=data["reason"],
                        messages=data["messages"],
                        pinned_files=data.get("pinned_files", []),
                        file_snapshots=data.get("file_snapshots", {}),
                        model=data.get("model", ""),
                        label=data.get("label", ""),
                    ))
            except Exception:
                continue
        return results

    def list_for_repo(self, repo_root: str, limit: int = 10) -> list[GhostSnapshot]:
        """List recent snapshots for a repo."""
        results = []
        for path in sorted(self.dir.glob("*.json"), reverse=True):
            if len(results) >= limit:
                break
            try:
                data = json.loads(path.read_text())
                if data.get("repo_root") == repo_root:
                    results.append(GhostSnapshot(
                        id=data["id"],
                        session_id=data["session_id"],
                        repo_root=data["repo_root"],
                        created_at=data["created_at"],
                        reason=data["reason"],
                        messages=data["messages"],
                        pinned_files=data.get("pinned_files", []),
                        file_snapshots=data.get("file_snapshots", {}),
                        model=data.get("model", ""),
                        label=data.get("label", ""),
                    ))
            except Exception:
                continue
        return results

    def delete(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        path = self.dir / f"{snapshot_id}.json"
        if path.is_file():
            path.unlink()
            return True
        return False

    def cleanup(self, max_age_days: int = 30) -> int:
        """Delete snapshots older than max_age_days."""
        cutoff = time.time() - (max_age_days * 86400)
        deleted = 0
        for path in self.dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if data.get("created_at", 0) < cutoff:
                    path.unlink()
                    deleted += 1
            except Exception:
                continue
        return deleted


def create_snapshot(session_id: str, repo_root: str, messages: list[dict],
                    pinned_files: list[str], model: str = "",
                    reason: str = "checkpoint", label: str = "",
                    capture_files: list[str] | None = None) -> GhostSnapshot:
    """Create a snapshot of the current conversation state.

    Args:
        capture_files: list of relative file paths to snapshot content for.
                      If None, no file content is captured.
    """
    file_snapshots: dict[str, str] = {}
    if capture_files:
        root = Path(repo_root)
        for rel_path in capture_files:
            full = root / rel_path
            if full.is_file():
                try:
                    file_snapshots[rel_path] = full.read_text(errors="replace")
                except Exception:
                    pass

    snapshot = GhostSnapshot(
        id=uuid.uuid4().hex[:10],
        session_id=session_id,
        repo_root=repo_root,
        created_at=time.time(),
        reason=reason,
        messages=list(messages),
        pinned_files=list(pinned_files),
        file_snapshots=file_snapshots,
        model=model,
        label=label,
    )

    store = SnapshotStore()
    store.save(snapshot)
    return snapshot


def restore_snapshot(snapshot: GhostSnapshot) -> dict:
    """Restore file system state from a snapshot.

    Returns dict of files restored.
    """
    restored = {}
    root = Path(snapshot.repo_root)
    for rel_path, content in snapshot.file_snapshots.items():
        full = root / rel_path
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            restored[rel_path] = "restored"
        except Exception as e:
            restored[rel_path] = f"error: {e}"
    return restored
