"""edit_diff — apply a unified diff patch via the system `patch(1)`."""
from __future__ import annotations

import os
import subprocess
import tempfile

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_diff",
        "description": (
            "Apply a unified diff (patch format) to a file. Input `diff` is "
            "the text of a `diff -u` / `git diff` hunk, starting with `@@` "
            "lines. Use when you want to express several non-adjacent "
            "changes as one coherent patch. Rejects cleanly on context "
            "mismatch — no partial application."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "diff": {"type": "string", "description": "Unified diff text, including @@ hunks"},
            },
            "required": ["path", "diff"],
        },
    },
}


def execute(ctx: ToolContext, args: dict) -> str:
    if "path" not in args or "diff" not in args:
        return "Error: 'path' and 'diff' (unified-diff text) required."
    path = ctx.resolve_path(args["path"])
    if not path.exists():
        return f"File not found: {args['path']}"

    diff_text = args["diff"]
    if not isinstance(diff_text, str) or not diff_text.strip():
        return "Error: 'diff' must be a non-empty unified-diff string."
    if not diff_text.endswith("\n"):
        diff_text += "\n"

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="lc-diff-", suffix=".patch")
        with os.fdopen(fd, "w") as f:
            f.write(diff_text)
        # Dry-run first so a bad context hunk doesn't half-apply.
        dry = subprocess.run(
            ["patch", "--dry-run", "--silent", "-u", str(path), "-i", tmp_path],
            capture_output=True, text=True, cwd=str(ctx.repo), timeout=15,
        )
        if dry.returncode != 0:
            return (
                f"Patch dry-run failed for {args['path']} "
                f"(exit {dry.returncode}):\n"
                f"{(dry.stderr or dry.stdout).strip()}"
            )
        real = subprocess.run(
            ["patch", "-u", str(path), "-i", tmp_path],
            capture_output=True, text=True, cwd=str(ctx.repo), timeout=15,
        )
        if real.returncode != 0:
            return (
                f"Patch apply failed for {args['path']} "
                f"(exit {real.returncode}):\n"
                f"{(real.stderr or real.stdout).strip()}"
            )
        added = sum(1 for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in diff_text.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        return f"Patched {args['path']}: +{added} lines, -{removed} lines."
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
