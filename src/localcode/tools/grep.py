"""grep — regex search across file contents."""
from __future__ import annotations

import subprocess

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": "Search file contents with regex. Returns matches with file paths and line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Directory to search. Default: repo root."},
                "include": {"type": "string", "description": "File glob filter, e.g. '*.py'"},
            },
            "required": ["pattern"],
        },
    },
}


def execute(ctx: ToolContext, args: dict) -> str:
    pattern = args["pattern"]
    repo = ctx.repo
    search_path = str(repo / args.get("path", "."))
    include = args.get("include", "")
    cmd = [
        "grep", "-rn",
        "--exclude-dir=.git",
        "--exclude-dir=.venv",
        "--exclude-dir=node_modules",
        "--exclude-dir=__pycache__",
    ]
    if include:
        cmd.append(f"--include={include}")
    cmd.extend([pattern, search_path])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        result = r.stdout.strip()
        result = result.replace(str(repo) + "/", "")
        lines = result.splitlines()
        if len(lines) > 50:
            result = "\n".join(lines[:50]) + f"\n\n[{len(lines) - 50} more matches]"
        return result or "No matches found"
    except subprocess.TimeoutExpired:
        return "Error: search timed out"


def is_concurrency_safe(args: dict) -> bool:
    return True
