"""Git worktree isolation — agent runs in a temp worktree so bad changes don't affect main."""
from __future__ import annotations

import subprocess
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Worktree:
    path: Path
    branch: str
    original_root: Path

    def has_changes(self) -> bool:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.path, capture_output=True, text=True, check=False,
        )
        return bool(result.stdout.strip())

    def diff_summary(self) -> str:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=self.path, capture_output=True, text=True, check=False,
        )
        return result.stdout.strip()


def create_worktree(repo_root: Path) -> Worktree | None:
    """Create a temporary git worktree for isolated agent work."""
    branch = f"jem-agent-{uuid.uuid4().hex[:8]}"
    worktree_path = repo_root.parent / f".jem-worktree-{branch}"

    try:
        # Create branch from current HEAD
        subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(worktree_path)],
            cwd=repo_root, capture_output=True, text=True, check=True, timeout=15,
        )
        return Worktree(path=worktree_path, branch=branch, original_root=repo_root)
    except Exception:
        return None


def merge_worktree(wt: Worktree) -> tuple[bool, str]:
    """Merge worktree changes back to main branch."""
    if not wt.has_changes():
        cleanup_worktree(wt)
        return True, "No changes to merge."

    try:
        # Commit changes in worktree
        subprocess.run(
            ["git", "add", "-A"], cwd=wt.path, check=True, capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", f"jem agent: {wt.branch}"],
            cwd=wt.path, check=True, capture_output=True, timeout=10,
        )
        # Merge back to original branch
        current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=wt.original_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        result = subprocess.run(
            ["git", "merge", wt.branch, "--no-edit"],
            cwd=wt.original_root, capture_output=True, text=True, check=False, timeout=30,
        )
        cleanup_worktree(wt)
        if result.returncode == 0:
            return True, f"Merged {wt.branch} into {current}."
        return False, f"Merge conflict:\n{result.stdout}\n{result.stderr}"
    except Exception as exc:
        return False, f"Merge failed: {exc}"


def discard_worktree(wt: Worktree) -> str:
    """Discard worktree changes entirely."""
    cleanup_worktree(wt)
    return f"Discarded {wt.branch}."


def cleanup_worktree(wt: Worktree) -> None:
    """Remove worktree and branch."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", str(wt.path), "--force"],
            cwd=wt.original_root, capture_output=True, check=False, timeout=10,
        )
    except Exception:
        if wt.path.exists():
            shutil.rmtree(wt.path, ignore_errors=True)
    try:
        subprocess.run(
            ["git", "branch", "-D", wt.branch],
            cwd=wt.original_root, capture_output=True, check=False, timeout=5,
        )
    except Exception:
        pass
