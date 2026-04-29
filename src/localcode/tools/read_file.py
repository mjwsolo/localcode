"""read_file — read a file with optional line offset/limit."""
from __future__ import annotations

from .base import ToolContext


DEFAULT_LIMIT = 240
MAX_DEFAULT_CHARS = 12_000

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file. You MUST read before editing. Returns content with line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "offset": {"type": "integer", "description": "Start line (0-based). Optional."},
                "limit": {
                    "type": "integer",
                    "description": (
                        "Max lines. Default 240. Use a small targeted range "
                        "for large files; only request a larger limit when the "
                        "whole file is genuinely needed."
                    ),
                },
            },
            "required": ["path"],
        },
    },
}


def execute(ctx: ToolContext, args: dict) -> str:
    if "path" not in args:
        return "Error: 'path' argument is required for read_file."
    path = ctx.repo / args["path"]
    if not path.exists():
        return f"File not found: {args['path']}"
    content = path.read_text(errors="replace")
    lines = content.splitlines()
    offset = args.get("offset", 0)
    explicit_limit = "limit" in args
    limit = args.get("limit", DEFAULT_LIMIT)
    selected = lines[offset:offset + limit]
    numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
    result = "\n".join(numbered)
    if len(lines) > offset + limit:
        result += f"\n\n[{len(lines) - offset - limit} more lines — use offset={offset + limit} to continue]"
    if not explicit_limit and len(result) > MAX_DEFAULT_CHARS:
        kept: list[str] = []
        total = 0
        for line in numbered:
            total += len(line) + 1
            if total > MAX_DEFAULT_CHARS:
                break
            kept.append(line)
        result = "\n".join(kept)
        remaining_from_line = offset + len(kept)
        result += (
            f"\n\n[Large file summarized at {MAX_DEFAULT_CHARS} chars. "
            f"File has {len(lines)} lines; continue with "
            f"offset={remaining_from_line}, limit={DEFAULT_LIMIT}, or request "
            "a focused smaller range around the symbol you need.]"
        )
    # Prompt-injection defence: wrap untrusted file content in explicit
    # data/instruction separator markers so the model knows this text
    # is DATA, not commands. Signature detector flags common injection
    # phrases (IGNORE ALL PRIOR INSTRUCTIONS, etc.) with a visible
    # warning before the content. See src/localcode/injection_defense.py.
    from ..injection_defense import wrap_untrusted
    return wrap_untrusted(result, source=str(args["path"]))


def is_concurrency_safe(args: dict) -> bool:
    return True
