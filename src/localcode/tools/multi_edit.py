"""multi_edit - apply N atomic edits to one file in a single call."""
from __future__ import annotations

import difflib
from dataclasses import dataclass

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "multi_edit",
        "description": (
            "Apply MULTIPLE edits to one file in a single call. Each edit is "
            "a (old_string, new_string) pair matched against the ORIGINAL "
            "file content, then applied atomically. Preferred "
            "over N sequential edit_file calls when refactoring — fewer "
            "round-trips, atomic success. Each old_string must be unique and "
            "must not overlap another edit. Keep anchors small but unique and "
            "do not pad edits with large unchanged regions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "edits": {
                    "type": "array",
                    "description": "Ordered list of edits to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
    },
}


@dataclass(frozen=True)
class _PreparedEdit:
    index: int
    start: int
    end: int
    old: str
    new: str


def execute(ctx: ToolContext, args: dict) -> str:
    if "path" not in args or "edits" not in args:
        return "Error: 'path' and 'edits' (list of {old_string, new_string}) required."
    path = ctx.repo / args["path"]
    if not path.exists():
        return f"File not found: {args['path']}"
    edits = args["edits"]
    if not isinstance(edits, list) or not edits:
        return "Error: 'edits' must be a non-empty list of {old_string, new_string} objects."

    original = path.read_text(errors="replace")
    prepared: list[_PreparedEdit] = []
    for i, ed in enumerate(edits, 1):
        if not isinstance(ed, dict):
            return f"Edit {i}: not an object; applied 0/{len(edits)}."
        old = ed.get("old_string", "")
        new = ed.get("new_string", "")
        if not isinstance(old, str) or not isinstance(new, str):
            return f"Edit {i}: old_string/new_string must be strings; applied 0/{len(edits)}."
        if not old:
            return f"Edit {i}: old_string is empty; applied 0/{len(edits)}."
        if old == new:
            return f"Edit {i}: no-op edit; old_string and new_string are identical. applied 0/{len(edits)}."
        starts = _find_all(original, old)
        if not starts:
            return f"Edit {i}: old_string not found in original content; applied 0/{len(edits)}."
        if len(starts) > 1:
            return f"Edit {i}: old_string matches {len(starts)} places in original content — must be unique. applied 0/{len(edits)}."
        start = starts[0]
        prepared.append(_PreparedEdit(i, start, start + len(old), old, new))

    prepared.sort(key=lambda item: item.start)
    for prev, curr in zip(prepared, prepared[1:]):
        if curr.start < prev.end:
            return (
                f"Edit {curr.index}: overlaps edit {prev.index}; applied 0/{len(edits)}. "
                "Merge nearby changes into one edit."
            )

    content = original
    for item in sorted(prepared, key=lambda item: item.start, reverse=True):
        content = content[:item.start] + item.new + content[item.end:]

    if content == original:
        return f"Error: no-op multi_edit on {args['path']}; applied 0/{len(edits)}."

    path.write_text(content)
    diff = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile=args["path"], tofile=args["path"], lineterm="",
    ))
    head = f"Applied {len(prepared)}/{len(edits)} edits to {args['path']}"
    if diff:
        return head + "\n" + "\n".join(diff[:120])
    return head


def _find_all(text: str, needle: str) -> list[int]:
    starts: list[int] = []
    pos = 0
    while True:
        idx = text.find(needle, pos)
        if idx < 0:
            return starts
        starts.append(idx)
        pos = idx + 1
