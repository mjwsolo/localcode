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
    path = ctx.resolve_write_path(args["path"])
    if not path.exists():
        return f"File not found: {args['path']}"
    edits = args["edits"]
    if not isinstance(edits, list) or not edits:
        return "Error: 'edits' must be a non-empty list of {old_string, new_string} objects."

    # ── Read-before-edit staleness guard ──
    # Refuse to edit a file the model hasn't fully read this session, or that
    # has changed on disk since it was read. See tools/read_state.py.
    try:
        from . import read_state
        _guard = read_state.guard_edit(ctx.app, path, args["path"])
    except Exception:
        _guard = None
    if _guard:
        return _guard

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

    # Clobber guard: an edit whose old_string is a substring of an EARLIER
    # edit's new_string means the edits step on each other — once the earlier
    # edit is applied, this edit's anchor would match text the earlier edit
    # just inserted, cascading an unintended change. A loud, specific error
    # beats silent corruption. (claude-code getPatchForEdits utils.ts:296-337.)
    # Newlines are trimmed off the anchor first so a trailing-newline sentinel
    # doesn't defeat the check.
    for i in range(len(edits)):
        old_i = str((edits[i] or {}).get("old_string", "")).strip("\n")
        if not old_i:
            continue
        for j in range(i):
            new_j = str((edits[j] or {}).get("new_string", ""))
            if old_i in new_j:
                return (
                    f"Edit {i + 1}: its old_string is a substring of edit "
                    f"{j + 1}'s new_string — these edits step on each other. "
                    f"applied 0/{len(edits)}. Reorder or merge them; do not "
                    "anchor one edit inside text another edit inserts."
                )

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
    # Refresh read-state so a follow-up edit on this path isn't blocked by the
    # read-before-edit guard (the model just wrote the current bytes).
    try:
        from . import read_state
        read_state.record_write(ctx.app, path, content)
    except Exception:
        pass
    _sw = ""
    try:
        from .syntax_check import check_syntax
        _e = check_syntax(str(path), content)
        if _e:
            _sw = f"\n\n⚠ SYNTAX ERROR after these edits — fix it now: {_e}"
    except Exception:
        _sw = ""
    diff = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile=args["path"], tofile=args["path"], lineterm="",
    ))
    head = f"Applied {len(prepared)}/{len(edits)} edits to {args['path']}"
    if diff:
        return head + "\n" + "\n".join(diff[:120]) + _sw
    return head + _sw


def _find_all(text: str, needle: str) -> list[int]:
    starts: list[int] = []
    pos = 0
    while True:
        idx = text.find(needle, pos)
        if idx < 0:
            return starts
        starts.append(idx)
        pos = idx + 1
