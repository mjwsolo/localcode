"""File watcher — detect external changes to tracked files.

Monitors pinned files and recently-read files for modifications
made outside of LocalCode (e.g., user editing in their IDE).
Notifies the user and invalidates cache.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileState:
    path: str
    mtime: float
    size: int


class FileWatcher:
    """Background thread that watches files for external changes."""

    def __init__(self, repo_root: Path, on_change: callable = None) -> None:
        self.repo_root = repo_root
        self._on_change = on_change
        self._tracked: dict[str, FileState] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._changes: list[str] = []  # paths changed since last check

    def track(self, relative_path: str) -> None:
        """Start watching a file."""
        abs_path = self.repo_root / relative_path
        if not abs_path.is_file():
            return
        try:
            stat = abs_path.stat()
            with self._lock:
                self._tracked[relative_path] = FileState(
                    path=relative_path,
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                )
        except Exception:
            pass

    def untrack(self, relative_path: str) -> None:
        with self._lock:
            self._tracked.pop(relative_path, None)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_changes(self) -> list[str]:
        """Get and clear list of changed file paths since last call."""
        with self._lock:
            changes = list(self._changes)
            self._changes.clear()
        return changes

    def _loop(self) -> None:
        while self._running:
            self._check()
            for _ in range(50):  # check every 5 seconds
                if not self._running:
                    return
                time.sleep(0.1)

    def _check(self) -> None:
        with self._lock:
            for rel_path, state in list(self._tracked.items()):
                abs_path = self.repo_root / rel_path
                try:
                    stat = abs_path.stat()
                    if stat.st_mtime != state.mtime or stat.st_size != state.size:
                        state.mtime = stat.st_mtime
                        state.size = stat.st_size
                        if rel_path not in self._changes:
                            self._changes.append(rel_path)
                except FileNotFoundError:
                    if rel_path not in self._changes:
                        self._changes.append(rel_path)


class ProjectWatcher:
    """Watch entire project for changes. Uses git for efficiency."""

    def __init__(self, repo_root: Path, poll_interval: float = 2.0) -> None:
        self.repo_root = repo_root
        self.interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._snapshot: dict[str, float] = {}  # path → mtime
        self._callbacks: list[callable] = []
        self._has_git = (repo_root / ".git").is_dir()

    def on_change(self, callback: callable) -> None:
        """Register callback: fn(changed: list[str], created: list[str], deleted: list[str])"""
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._running:
            return
        self._snapshot = self._take_snapshot()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        while self._running:
            try:
                new_snapshot = self._take_snapshot()
                changes = self._diff(self._snapshot, new_snapshot)
                if any(changes.values()):
                    for cb in self._callbacks:
                        try:
                            cb(changes["modified"], changes["created"], changes["deleted"])
                        except Exception:
                            pass
                self._snapshot = new_snapshot
            except Exception:
                pass
            # Sleep in small increments for responsive shutdown
            for _ in range(int(self.interval * 10)):
                if not self._running:
                    return
                time.sleep(0.1)

    def _take_snapshot(self) -> dict[str, float]:
        """Get mtime for tracked files. Uses git ls-files for speed."""
        snapshot: dict[str, float] = {}
        if self._has_git:
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "ls-files"],
                    capture_output=True, text=True, timeout=5,
                    cwd=str(self.repo_root),
                )
                for line in result.stdout.splitlines():
                    fpath = self.repo_root / line
                    try:
                        snapshot[line] = fpath.stat().st_mtime
                    except (OSError, FileNotFoundError):
                        pass
                return snapshot
            except Exception:
                pass

        # Fallback: walk directory
        skip = {".git", "node_modules", "__pycache__", "venv", ".venv"}
        for p in self.repo_root.rglob("*"):
            if p.is_file() and not any(s in p.parts for s in skip):
                rel = str(p.relative_to(self.repo_root))
                try:
                    snapshot[rel] = p.stat().st_mtime
                except OSError:
                    pass
        return snapshot

    @staticmethod
    def _diff(old: dict[str, float], new: dict[str, float]) -> dict[str, list[str]]:
        old_keys = set(old)
        new_keys = set(new)
        return {
            "created": list(new_keys - old_keys),
            "deleted": list(old_keys - new_keys),
            "modified": [p for p in old_keys & new_keys if old[p] != new[p]],
        }
