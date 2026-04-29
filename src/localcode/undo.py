"""LocalCode undo — tracks file changes and supports rollback."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class FileSnapshot:
    """A before-snapshot of a file before it was changed."""
    path: str           # relative to repo root
    existed: bool       # whether the file existed before
    content: str        # original content (empty if didn't exist)
    timestamp: float
    tool_name: str      # which tool made the change


@dataclass
class ChangeLog:
    """Ordered log of file changes in a session, supporting undo."""
    repo_root: Path
    snapshots: list[FileSnapshot] = field(default_factory=list)

    def snapshot_before(self, relative_path: str, tool_name: str = "") -> None:
        """Take a snapshot of a file BEFORE it gets modified."""
        abs_path = (self.repo_root / relative_path).resolve()
        existed = abs_path.is_file()
        content = ""
        if existed:
            try:
                content = abs_path.read_text(errors="replace")
            except Exception:
                pass
        self.snapshots.append(FileSnapshot(
            path=relative_path,
            existed=existed,
            content=content,
            timestamp=time.time(),
            tool_name=tool_name,
        ))

    def undo_last(self) -> tuple[bool, str]:
        """Undo the most recent file change. Returns (success, message)."""
        if not self.snapshots:
            return False, "Nothing to undo."

        snapshot = self.snapshots.pop()
        abs_path = (self.repo_root / snapshot.path).resolve()

        if not snapshot.existed:
            # File was created — delete it
            if abs_path.exists():
                abs_path.unlink()
                return True, f"Deleted {snapshot.path} (was created by {snapshot.tool_name})"
            return True, f"File already gone: {snapshot.path}"

        # File existed before — restore original content
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(snapshot.content)
            return True, f"Restored {snapshot.path} (changed by {snapshot.tool_name})"
        except Exception as exc:
            return False, f"Failed to restore {snapshot.path}: {exc}"

    def undo_all(self) -> list[str]:
        """Undo all changes in reverse order. Returns list of messages."""
        messages = []
        while self.snapshots:
            ok, msg = self.undo_last()
            messages.append(msg)
        return messages

    def status(self) -> list[dict[str, str]]:
        """List all tracked changes."""
        rows = []
        for snap in self.snapshots:
            action = "created" if not snap.existed else "modified"
            ts = time.strftime("%H:%M:%S", time.localtime(snap.timestamp))
            rows.append({
                "path": snap.path,
                "action": action,
                "tool": snap.tool_name,
                "time": ts,
            })
        return rows

    @property
    def change_count(self) -> int:
        return len(self.snapshots)

    def recent_files(self, since: int = 0) -> list[str]:
        """Get list of files changed since a given snapshot index."""
        seen = set()
        result = []
        for snap in self.snapshots[since:]:
            if snap.path not in seen:
                seen.add(snap.path)
                result.append(snap.path)
        return result

    def start_transaction(self) -> int:
        """Mark the start of a multi-file edit transaction. Returns transaction ID."""
        return len(self.snapshots)

    def undo_transaction(self, transaction_start: int) -> list[str]:
        """Undo all changes since transaction_start."""
        messages = []
        while len(self.snapshots) > transaction_start:
            ok, msg = self.undo_last()
            messages.append(msg)
        return messages
