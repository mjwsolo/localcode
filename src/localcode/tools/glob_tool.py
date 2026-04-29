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
    results = [
        str(m.relative_to(repo)) for m in matches[:100]
        if not any(p in str(m) for p in (".git", "__pycache__", "node_modules", ".venv"))
    ]
    return "\n".join(results) or "No matches"


def is_concurrency_safe(args: dict) -> bool:
    return True
