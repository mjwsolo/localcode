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
    pattern = args["pattern"]
    repo = ctx.repo
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
    ][:100]
    return "\n".join(results) or "No matches"


def is_concurrency_safe(args: dict) -> bool:
    return True
