from __future__ import annotations

from pathlib import Path
import subprocess

from ._subproc_env import clean_env


IGNORE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
}


# Non-.git project markers, in the same spirit as agent/context._PROJECT_MARKERS.
# When a directory has no git history, one of these still identifies the ROOT of
# a real project — so a deep build dir anchors to its own project instead of
# silently adopting $HOME.
_ROOT_PROJECT_MARKERS = (
    "package.json",
    "pyproject.toml",
    "go.mod",
    "Cargo.toml",
    "tsconfig.json",
)


def _is_home_or_shallower(path: Path) -> bool:
    """True when `path` is $HOME, its parent, or the filesystem root/anchor.

    These are never legitimate project roots. Adopting one forces the model to
    carry long absolute paths (e.g. `$HOME/Desktop/Github/.../Anki/<model>/…`)
    on every tool call, which it then corrupts (`Aki`, `gitHub`, hallucinated
    users) and dies in failed-read loops. This predicate lets us refuse them.
    """
    try:
        home = Path.home().resolve()
    except (RuntimeError, OSError):
        return False
    return path in {home, home.parent, Path(path.anchor)}


def find_repo_root(start: Path) -> Path:
    """Locate the project root to anchor the model's paths against.

    Order of preference:
      1. The nearest ancestor containing `.git` — a real VCS root. This path is
         behaviour-IDENTICAL to the original implementation and is what runs
         whenever the user is inside a git checkout.
      2. If there is no `.git` anywhere above `start`, the nearest ancestor
         carrying a non-git project marker (package.json / pyproject.toml /
         go.mod / Cargo.toml / tsconfig.json). This pins a *real* project dir
         instead of walking all the way up to (and silently adopting) $HOME.
      3. Otherwise `start` itself — but NEVER bare $HOME (or shallower). When
         `start` resolves to $HOME with no project marker, we still hand back
         $HOME (there is nothing nearer to anchor to), but callers can detect
         that degenerate case via `_is_home_or_shallower` and mark it.
    """
    current = start.resolve()
    chain = [current, *current.parents]

    # 1. A real .git root always wins — identical to the original behaviour.
    for candidate in chain:
        if (candidate / ".git").exists():
            return candidate

    # 2. No .git: prefer the nearest non-git project marker, but never accept a
    #    marker at $HOME or shallower (those are not project roots).
    for candidate in chain:
        if _is_home_or_shallower(candidate):
            break
        if any((candidate / marker).exists() for marker in _ROOT_PROJECT_MARKERS):
            return candidate

    # 3. Nothing better than the launch dir. Return it unchanged (this is cwd);
    #    if it IS $HOME the caller is responsible for marking the degenerate case.
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
            env=clean_env(),
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
            env=clean_env(),
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
