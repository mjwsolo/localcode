"""glob — find files matching a glob pattern (filename is glob_tool.py to avoid
shadowing stdlib `glob`)."""
from __future__ import annotations

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
            },
            "required": ["pattern"],
        },
    },
}


def execute(ctx: ToolContext, args: dict) -> str:
    pattern = str(args["pattern"])
    repo = ctx.repo
    # Containment: a pattern like `../../home/*/.ssh/*` or an absolute path
    # would enumerate files OUTSIDE the repo — the read-side twin of the
    # write-containment guard. Reject up-front, and belt-and-braces filter
    # every match (symlinks) back inside the resolved repo root.
    if pattern.startswith(("/", "~")) or ".." in pattern.split("/"):
        return "REJECTED: glob patterns must stay inside the repo (no leading '/' or '..' segments)."
    try:
        repo_resolved = repo.resolve()
    except OSError:
        repo_resolved = repo
    matches = sorted(
        repo.glob(pattern),
        key=lambda f: f.stat().st_mtime if f.exists() else 0,
        reverse=True,
    )
    # Exclude noise dirs BEFORE truncating — otherwise a repo whose 100 newest
    # matches are dominated by node_modules/.venv would hide real source files
    # that sort just past the cut.
    results = [
        str(m.relative_to(repo)) for m in matches
        if not any(p in str(m) for p in (".git", "__pycache__", "node_modules", ".venv"))
        and _contained(m, repo_resolved)
    ][:100]
    return "\n".join(results) or "No matches"


def _contained(path, repo_resolved) -> bool:
    try:
        return path.resolve().is_relative_to(repo_resolved)
    except OSError:
        return False


def is_concurrency_safe(args: dict) -> bool:
    return True
