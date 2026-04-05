from __future__ import annotations

from pathlib import Path
import subprocess


IGNORE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
}


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return current


def list_repo_files(repo_root: Path, pattern: str | None = None, limit: int = 200) -> list[str]:
    files: list[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(repo_root))
        if pattern and pattern.lower() not in rel.lower():
            continue
        files.append(rel)
        if len(files) >= limit:
            break
    return files


def read_file(repo_root: Path, relative_path: str, max_chars: int = 12000) -> str:
    path = (repo_root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(relative_path)
    content = path.read_text(errors="replace")
    if len(content) > max_chars:
        return content[:max_chars] + "\n...[truncated]"
    return content


def git_status(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "clean"
    except FileNotFoundError:
        return "git unavailable"


def git_diff(repo_root: Path, max_chars: int = 20000) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        diff = result.stdout.strip()
        if not diff:
            return "No working tree diff."
        if len(diff) > max_chars:
            return diff[:max_chars] + "\n...[truncated]"
        return diff
    except FileNotFoundError:
        return "git unavailable"


def build_context_block(repo_root: Path, pinned_files: list[str], max_chars: int) -> str:
    sections: list[str] = [
        f"Repository root: {repo_root}",
        "Git status:",
        git_status(repo_root),
    ]
    for relative_path in pinned_files:
        try:
            content = read_file(repo_root, relative_path, max_chars=max(2000, max_chars // max(1, len(pinned_files))))
            sections.append(f"File: {relative_path}\n```text\n{content}\n```")
        except FileNotFoundError:
            sections.append(f"File missing: {relative_path}")
    text = "\n\n".join(sections)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text
