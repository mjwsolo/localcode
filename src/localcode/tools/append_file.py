"""append_file - append content to a text file, creating it when needed."""
from __future__ import annotations

from .base import ToolContext
from .write_file import _detect_stub_code


SCHEMA = {
    "type": "function",
    "function": {
        "name": "append_file",
        "description": (
            "Append content to a text file, creating the file and parent "
            "directories if needed. Use this for section-by-section file "
            "creation or when adding a later section without an edit anchor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {
                    "type": "string",
                    "description": "Content to append to the end of the file",
                },
            },
            "required": ["path", "content"],
        },
    },
}


def execute(ctx: ToolContext, args: dict) -> str:
    if "path" not in args:
        return "Error: 'path' argument is required for append_file."
    if "content" not in args:
        return "Error: 'content' argument is required for append_file."

    rel_path = str(args["path"])
    path = ctx.resolve_write_path(rel_path)
    if path.is_dir():
        return f"Error: append_file cannot append to directory: {rel_path}"
    content = str(args.get("content", ""))
    stub = _detect_stub_code(content, rel_path)
    if stub:
        return stub

    created = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = "" if created else path.read_text(errors="replace")
    separator = "" if not existing or existing.endswith("\n") or content.startswith("\n") else "\n"
    path.write_text(existing + separator + content)
    appended_lines = content.count("\n") + (1 if content else 0)
    total_lines = (existing + separator + content).count("\n") + 1
    verb = "Created" if created else "Appended"
    return f"{verb} {rel_path}; added {appended_lines} lines ({total_lines} total lines)"
