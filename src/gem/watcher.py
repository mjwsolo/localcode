"""File watcher — detect external changes to tracked files.

Monitors pinned files and recently-read files for modifications
made outside of Jem (e.g., user editing in their IDE).
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
