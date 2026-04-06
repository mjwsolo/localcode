"""Diff engine — preview, approve, and apply file changes.

Provides the safety layer between model output and disk writes:
1. Generate colored unified diffs
2. Show preview to user
3. Get approval (y/n/a/e)
4. Apply with undo support
5. Batch preview for multi-file edits

Integrates with the UndoStack in context_manager.py.
"""
from __future__ import annotations

import difflib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── ANSI colors ─────────────────────────────────────────────────────

RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
YELLOW = "\033[93m"
RESET = "\033[0m"


@dataclass
class EditAction:
    """A single edit to apply to a file."""
    file_path: str
    edit_type: str  # "search_replace" | "full_write" | "create"
    old_string: str = ""
    new_string: str = ""
    full_content: str = ""


@dataclass
class EditResult:
    """Result of an edit action."""
    file_path: str
    applied: bool
    reason: str = ""  # "applied", "rejected", "error: ..."
    lines_added: int = 0
    lines_removed: int = 0


class DiffEngine:
    """Generate diffs, get approval, apply changes.

    Usage:
        engine = DiffEngine(project_root, undo_stack)
        result = engine.preview_and_apply(edit_action)
        # or for batch:
        results = engine.preview_and_apply_batch(actions)
    """

    def __init__(self, project_root: str | Path,
                 undo_stack: Any = None,
                 auto_approve: bool = False,
                 max_diff_lines: int = 40) -> None:
        self.root = Path(project_root)
        self.undo = undo_stack
        self.auto_approve = auto_approve
        self.max_diff_lines = max_diff_lines

    # ── Single edit ─────────────────────────────────────────────────

    def preview_and_apply(self, action: EditAction) -> EditResult:
        """Show diff, get approval, apply if approved."""
        full_path = self.root / action.file_path

        if action.edit_type == "create":
            return self._handle_create(action, full_path)
        elif action.edit_type == "full_write":
            return self._handle_full_write(action, full_path)
        elif action.edit_type == "search_replace":
            return self._handle_search_replace(action, full_path)
        else:
            return EditResult(action.file_path, False, f"unknown edit type: {action.edit_type}")

    # ── Batch edit ──────────────────────────────────────────────────

    def preview_and_apply_batch(self, actions: list[EditAction]) -> list[EditResult]:
        """Preview and apply multiple edits with apply-all support."""
        results = []
        apply_all = self.auto_approve

        for i, action in enumerate(actions):
            header = f"{BOLD}Edit {i + 1}/{len(actions)}: {action.file_path}{RESET}"
            sys.stdout.write(f"\n{header}\n")

            if apply_all:
                result = self._apply_without_preview(action)
                results.append(result)
                if result.applied:
                    sys.stdout.write(f"{GREEN}  ✓ Applied{RESET}\n")
                else:
                    sys.stdout.write(f"{RED}  ✗ {result.reason}{RESET}\n")
                continue

            result = self.preview_and_apply(action)
            results.append(result)

            # Check if user said "apply all"
            if result.reason == "apply_all":
                apply_all = True

        return results

    # ── Diff generation ─────────────────────────────────────────────

    def generate_diff(self, old_content: str, new_content: str,
                      file_path: str = "") -> list[str]:
        """Generate unified diff lines."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{file_path}" if file_path else "a/file",
            tofile=f"b/{file_path}" if file_path else "b/file",
            lineterm="",
        ))
        return diff

    def format_diff_colored(self, diff_lines: list[str],
                            max_lines: int | None = None) -> str:
        """Format diff with ANSI colors."""
        if max_lines is None:
            max_lines = self.max_diff_lines

        lines = []
        shown = 0
        total = len(diff_lines)

        for line in diff_lines:
            if shown >= max_lines:
                remaining = total - shown
                lines.append(f"{DIM}  ... +{remaining} more lines{RESET}")
                break

            if line.startswith("+++") or line.startswith("---"):
                lines.append(f"  {BOLD}{line}{RESET}")
            elif line.startswith("@@"):
                lines.append(f"  {CYAN}{line}{RESET}")
            elif line.startswith("+"):
                lines.append(f"  {GREEN}{line}{RESET}")
            elif line.startswith("-"):
                lines.append(f"  {RED}{line}{RESET}")
            else:
                lines.append(f"  {DIM}{line}{RESET}")
            shown += 1

        return "\n".join(lines)

    def diff_stats(self, diff_lines: list[str]) -> tuple[int, int]:
        """Count additions and removals in a diff."""
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        return added, removed

    # ── Handlers ────────────────────────────────────────────────────

    def _handle_create(self, action: EditAction, full_path: Path) -> EditResult:
        """Handle creating a new file."""
        content = action.full_content
        lines = content.splitlines()
        line_count = len(lines)

        # Show preview
        sys.stdout.write(f"\n{BOLD}  Create new file: {action.file_path}{RESET}\n")
        preview_lines = min(20, line_count)
        for i, line in enumerate(lines[:preview_lines], 1):
            sys.stdout.write(f"  {GREEN}+{i:4d} │ {line}{RESET}\n")
        if line_count > preview_lines:
            sys.stdout.write(f"  {DIM}  ... +{line_count - preview_lines} more lines{RESET}\n")
        sys.stdout.write(f"\n  {DIM}{line_count} lines total{RESET}\n\n")
        sys.stdout.flush()

        # Get approval
        if self.auto_approve:
            decision = "y"
        else:
            decision = self._ask_approval()

        if decision in ("n", "reject"):
            return EditResult(action.file_path, False, "rejected")
        if decision == "a":
            self._snapshot(action.file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return EditResult(action.file_path, True, "apply_all",
                              lines_added=line_count, lines_removed=0)

        # Apply
        self._snapshot(action.file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return EditResult(action.file_path, True, "applied",
                          lines_added=line_count, lines_removed=0)

    def _handle_full_write(self, action: EditAction, full_path: Path) -> EditResult:
        """Handle overwriting an existing file."""
        if not full_path.is_file():
            # File doesn't exist — treat as create
            action.edit_type = "create"
            return self._handle_create(action, full_path)

        old_content = full_path.read_text(errors="replace")
        new_content = action.full_content

        if old_content == new_content:
            return EditResult(action.file_path, False, "no changes")

        # Generate and show diff
        diff = self.generate_diff(old_content, new_content, action.file_path)
        added, removed = self.diff_stats(diff)

        sys.stdout.write(f"\n{BOLD}  Overwrite: {action.file_path}{RESET}\n")
        sys.stdout.write(self.format_diff_colored(diff) + "\n")
        sys.stdout.write(f"\n  {DIM}-{removed} +{added} lines{RESET}\n\n")
        sys.stdout.flush()

        # Get approval
        if self.auto_approve:
            decision = "y"
        else:
            decision = self._ask_approval()

        if decision in ("n", "reject"):
            return EditResult(action.file_path, False, "rejected")

        reason = "apply_all" if decision == "a" else "applied"
        self._snapshot(action.file_path)
        full_path.write_text(new_content)
        return EditResult(action.file_path, True, reason,
                          lines_added=added, lines_removed=removed)

    def _handle_search_replace(self, action: EditAction, full_path: Path) -> EditResult:
        """Handle a search/replace edit."""
        if not full_path.is_file():
            return EditResult(action.file_path, False, f"file not found: {action.file_path}")

        content = full_path.read_text(errors="replace")

        # Validate search string exists
        count = content.count(action.old_string)
        if count == 0:
            return EditResult(action.file_path, False,
                              "error: search string not found in file")
        if count > 1:
            return EditResult(action.file_path, False,
                              f"error: search string matches {count} locations (ambiguous)")

        # Generate diff
        new_content = content.replace(action.old_string, action.new_string, 1)
        diff = self.generate_diff(content, new_content, action.file_path)
        added, removed = self.diff_stats(diff)

        sys.stdout.write(f"\n{BOLD}  Edit: {action.file_path}{RESET}\n")
        sys.stdout.write(self.format_diff_colored(diff) + "\n")
        sys.stdout.write(f"\n  {DIM}-{removed} +{added}{RESET}\n\n")
        sys.stdout.flush()

        # Get approval
        if self.auto_approve:
            decision = "y"
        else:
            decision = self._ask_approval()

        if decision in ("n", "reject"):
            return EditResult(action.file_path, False, "rejected")

        reason = "apply_all" if decision == "a" else "applied"
        self._snapshot(action.file_path)
        full_path.write_text(new_content)
        return EditResult(action.file_path, True, reason,
                          lines_added=added, lines_removed=removed)

    # ── Approval prompt ─────────────────────────────────────────────

    @staticmethod
    def _ask_approval() -> str:
        """Get single-keypress approval from user.

        Returns: 'y' (yes), 'n' (no), 'a' (all remaining), 'e' (edit)
        """
        sys.stdout.write(f"  {YELLOW}Apply? [y]es / [n]o / [a]ll remaining: {RESET}")
        sys.stdout.flush()

        try:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write(f"{ch}\n")
            sys.stdout.flush()
        except Exception:
            ch = input().strip().lower()[:1] or "y"

        return {
            "y": "y", "\r": "y", "\n": "y", "": "y",
            "n": "n",
            "a": "a",
            "e": "e",
        }.get(ch, "n")

    # ── Undo support ────────────────────────────────────────────────

    def _snapshot(self, rel_path: str) -> None:
        """Take a snapshot before modifying a file."""
        if self.undo:
            try:
                self.undo.snapshot(str(self.root / rel_path))
            except Exception:
                pass

    def _apply_without_preview(self, action: EditAction) -> EditResult:
        """Apply an edit without showing diff (for apply-all mode)."""
        full_path = self.root / action.file_path

        if action.edit_type == "create":
            self._snapshot(action.file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(action.full_content)
            return EditResult(action.file_path, True, "applied",
                              lines_added=len(action.full_content.splitlines()))

        elif action.edit_type == "full_write":
            self._snapshot(action.file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(action.full_content)
            return EditResult(action.file_path, True, "applied")

        elif action.edit_type == "search_replace":
            if not full_path.is_file():
                return EditResult(action.file_path, False, "file not found")
            content = full_path.read_text(errors="replace")
            if action.old_string not in content:
                return EditResult(action.file_path, False, "search string not found")
            self._snapshot(action.file_path)
            full_path.write_text(content.replace(action.old_string, action.new_string, 1))
            return EditResult(action.file_path, True, "applied")

        return EditResult(action.file_path, False, "unknown edit type")


# ── Convenience functions ───────────────────────────────────────────

def quick_diff(old: str, new: str, filename: str = "file") -> str:
    """Generate a quick colored diff string for display."""
    engine = DiffEngine(".")
    diff_lines = engine.generate_diff(old, new, filename)
    return engine.format_diff_colored(diff_lines, max_lines=30)


def show_file_diff(file_path: str, new_content: str, project_root: str = ".") -> str:
    """Show diff between current file and proposed new content."""
    full = Path(project_root) / file_path
    if full.is_file():
        old = full.read_text(errors="replace")
    else:
        old = ""
    return quick_diff(old, new_content, file_path)
