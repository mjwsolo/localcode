"""Turn-level diff aggregation — show what changed in one user turn.

Instead of showing individual file operation diffs, aggregate all changes
since the turn started into one unified view:

    Turn summary:
      +15 -3  auth.py (modified)
      +42     models/user.py (created)
      -1      old_config.py (deleted)
    Total: 3 files, +57 -4

This gives users a clear "what did the agent do" picture.
"""
from __future__ import annotations

import difflib
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileChange:
    """A single file's changes within a turn."""
    path: str
    action: str  # "created" | "modified" | "deleted"
    lines_added: int = 0
    lines_removed: int = 0
    old_content: str = ""
    new_content: str = ""


@dataclass
class TurnDiff:
    """Aggregated changes for an entire turn."""
    changes: list[FileChange] = field(default_factory=list)

    @property
    def total_added(self) -> int:
        return sum(c.lines_added for c in self.changes)

    @property
    def total_removed(self) -> int:
        return sum(c.lines_removed for c in self.changes)

    @property
    def files_changed(self) -> int:
        return len(self.changes)

    @property
    def file_paths(self) -> list[str]:
        return [c.path for c in self.changes]


class TurnDiffTracker:
    """Track file changes within a single user turn.

    Usage:
        tracker = TurnDiffTracker(repo_root)
        tracker.start_turn()
        # ... agent does work ...
        diff = tracker.end_turn()
        print(format_turn_diff(diff))
    """

    def __init__(self, repo_root: str | Path) -> None:
        self.root = Path(repo_root)
        self._baselines: dict[str, str | None] = {}  # path → content at turn start (None = didn't exist)
        self._tracking = False

    def start_turn(self, watched_files: list[str] | None = None) -> None:
        """Snapshot current state of tracked files at turn start."""
        self._baselines.clear()
        self._tracking = True

        if watched_files:
            for rel_path in watched_files:
                self._snapshot(rel_path)

    def track_file(self, rel_path: str) -> None:
        """Start tracking a file mid-turn (called when agent reads/writes a file)."""
        if self._tracking and rel_path not in self._baselines:
            self._snapshot(rel_path)

    def end_turn(self) -> TurnDiff:
        """Compute aggregated diff for the turn."""
        self._tracking = False
        changes = []

        for rel_path, old_content in self._baselines.items():
            full = self.root / rel_path
            new_exists = full.is_file()

            if old_content is None and new_exists:
                # File was created
                new_content = full.read_text(errors="replace")
                added = len(new_content.splitlines())
                changes.append(FileChange(
                    path=rel_path, action="created",
                    lines_added=added, lines_removed=0,
                    old_content="", new_content=new_content,
                ))

            elif old_content is not None and not new_exists:
                # File was deleted
                removed = len(old_content.splitlines())
                changes.append(FileChange(
                    path=rel_path, action="deleted",
                    lines_added=0, lines_removed=removed,
                    old_content=old_content, new_content="",
                ))

            elif old_content is not None and new_exists:
                # File was (potentially) modified
                new_content = full.read_text(errors="replace")
                if new_content != old_content:
                    old_lines = old_content.splitlines()
                    new_lines = new_content.splitlines()
                    diff = list(difflib.unified_diff(old_lines, new_lines))
                    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
                    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
                    changes.append(FileChange(
                        path=rel_path, action="modified",
                        lines_added=added, lines_removed=removed,
                        old_content=old_content, new_content=new_content,
                    ))

        self._baselines.clear()
        return TurnDiff(changes=changes)

    def _snapshot(self, rel_path: str) -> None:
        """Take a baseline snapshot of a file."""
        full = self.root / rel_path
        if full.is_file():
            try:
                self._baselines[rel_path] = full.read_text(errors="replace")
            except Exception:
                self._baselines[rel_path] = ""
        else:
            self._baselines[rel_path] = None  # doesn't exist yet


# ── Display ─────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_turn_diff(diff: TurnDiff) -> str:
    """Format a TurnDiff for terminal display."""
    if not diff.changes:
        return ""

    lines = [f"\n{BOLD}  Turn summary:{RESET}"]

    for change in diff.changes:
        added = f"{GREEN}+{change.lines_added}{RESET}" if change.lines_added else ""
        removed = f"{RED}-{change.lines_removed}{RESET}" if change.lines_removed else ""
        stats = f"{added} {removed}".strip()
        action = f"{DIM}({change.action}){RESET}"
        lines.append(f"    {stats:>12}  {change.path} {action}")

    # Total
    lines.append(
        f"  {DIM}Total: {diff.files_changed} file(s), "
        f"{GREEN}+{diff.total_added}{RESET} {RED}-{diff.total_removed}{RESET}{DIM}{RESET}"
    )

    return "\n".join(lines)


def print_turn_diff(diff: TurnDiff) -> None:
    """Print turn diff to stdout."""
    text = format_turn_diff(diff)
    if text:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
