"""list_files — directory listing with hidden/build-artifact filtering."""
from __future__ import annotations

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List directory contents. Skips hidden files and build artifacts.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path. Default: repo root."},
            },
        },
    },
}


_SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", "dist", "build"}


def execute(ctx: ToolContext, args: dict) -> str:
    target = ctx.repo / args.get("path", ".")
    if not target.exists():
        return f"Directory not found: {args.get('path', '.')}"
    entries = []
    for p in sorted(target.iterdir()):
        if p.name.startswith(".") and p.name != ".env":
            continue
        if p.name in _SKIP:
            continue
        marker = "/" if p.is_dir() else ""
        entries.append(f"  {p.name}{marker}")
    return "\n".join(entries[:200]) or "Empty directory"


def is_concurrency_safe(args: dict) -> bool:
    return True
