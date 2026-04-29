"""Tool execution guards and bookkeeping for the agent loop."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any


@dataclass
class ToolExecutionState:
    files_read: dict[str, int] = field(default_factory=dict)
    files_modified: set[str] = field(default_factory=set)
    io_calls: dict[tuple[str, str], int] = field(default_factory=dict)
    bash_failures: dict[str, int] = field(default_factory=dict)
    failed_calls: dict[tuple[str, str], int] = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    # Successful (tool_name, canonical_args) repeats — covers the gap
    # left when only `failed_calls` and `bash_failures` were tracked.
    # The 3-in-a-row exact-repeat guard was retired 2026-04-26 with the
    # rationale "fired 0 times in observed sessions"; on 2026-04-29 a
    # weather lookup ran the same `curl wttr.in/Paris?format=…` four
    # times in a row and nothing stopped it. Telemetry behind the
    # removal didn't include info-fetch loops.
    success_counts: dict[tuple[str, str], int] = field(default_factory=dict)


def canonical_args(args: dict[str, Any]) -> str:
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(args)


def dedup_stub_for_tool(
    tool_name: str,
    args: dict[str, Any],
    state: ToolExecutionState,
) -> str | None:
    if tool_name == "bash":
        cmd_key = re.sub(r"\s+", " ", str(args.get("command", "") or "").strip())
        failures = state.bash_failures.get(cmd_key, 0)
        if cmd_key and failures >= 2:
            return (
                f"REJECTED: this exact bash command already failed {failures} "
                "times this turn. Do not retry it unchanged. Read the error, "
                "inspect the relevant config/file, or choose a different command."
            )
    elif tool_name == "read_file":
        # No DEDUP for read_file. Earlier we returned a "[DEDUP …]" stub
        # whenever the model re-read a path it had already read in this
        # turn. Intent: discourage useless re-reads to save context.
        # Real cost (observed 2026-04-29): when the model is in a
        # legitimate debugging loop — verification probe failed → model
        # tries to re-inspect the source to fix the gap — the dedup
        # shield starves it of bytes for round after round, producing a
        # 17-minute hang on a single turn (5 read_file calls returning
        # the stub, then a grep, then more reads). The dedup logic
        # cannot tell "model is repeating itself uselessly" from "model
        # just learned new info from a probe and needs to re-look at the
        # file." Re-reads are cheap; recovery from a starved debug loop
        # is not. Trust the model to re-read when it needs to.
        pass
    elif tool_name in {"list_files", "glob", "grep"}:
        key = (tool_name, canonical_args(args))
        if key in state.io_calls:
            return (
                f"[DEDUP - same {tool_name} args as round {state.io_calls[key]} "
                "of this turn. No writes have happened since; the result is "
                "identical. Use the earlier output. If you need a refreshed "
                "view, write/edit a file first or call a different tool.]"
            )
    elif tool_name in {"write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}:
        key = (tool_name, canonical_args(args))
        failures = state.failed_calls.get(key, 0)
        if failures >= 2:
            return (
                f"REJECTED: this exact {tool_name} call already failed "
                f"{failures} times this turn. Do not retry it unchanged. "
                "Use a different tool or a materially different edit: read "
                "the current file state, choose a smaller exact anchor, use "
                "multi_edit/edit_diff where appropriate, or move on if the "
                "file is already correct."
            )
    return None


_REPEAT_STUB_TOOLS: frozenset[str] = frozenset({
    "bash", "web_fetch", "web_search", "launch_app",
})
_REPEAT_STUB_THRESHOLD = 2  # 3rd identical call fires the stub


def repeat_stub_for_tool(
    tool_name: str,
    args: dict[str, Any],
    state: ToolExecutionState,
) -> str | None:
    """Return a REJECTED stub when the same (tool_name, args) has
    already succeeded twice this turn. Covers tools without their own
    dedup path: bash / web_fetch / web_search / launch_app. read_file
    is intentionally exempt (re-reads after probes are legitimate);
    list_files / glob / grep are handled by `dedup_stub_for_tool`;
    write_file / edit_file / multi_edit / edit_diff are handled by
    `failed_calls`. Threshold is 2 prior successes — the 3rd identical
    call gets the stub instead of executing.
    """
    if tool_name not in _REPEAT_STUB_TOOLS:
        return None
    key = (tool_name, canonical_args(args))
    count = state.success_counts.get(key, 0)
    if count < _REPEAT_STUB_THRESHOLD:
        return None
    if tool_name == "bash":
        cmd_preview = str(args.get("command", ""))[:120].replace("\n", " ")
        return (
            f"REJECTED: this exact bash command already ran {count} times "
            f"this turn (`{cmd_preview}`) and produced the same output. "
            "Use the previous result, or run a materially different "
            "command. Do not retry the same one."
        )
    return (
        f"REJECTED: this exact {tool_name} call already ran {count} times "
        "this turn with the same arguments. Use the prior result or "
        "change strategy."
    )


def oversize_stub_for_tool(tool_name: str, args: dict[str, Any], max_bytes: int) -> str | None:
    try:
        arg_bytes = len(json.dumps(args, default=str))
    except Exception:
        arg_bytes = 0
    if (
        tool_name not in {"write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}
        or arg_bytes <= max_bytes
    ):
        return None
    path_hint = args.get("path") or args.get("file_path") or "<unknown>"
    return (
        f"REJECTED: this {tool_name} call is {arg_bytes:,} bytes - over the "
        f"{max_bytes:,} byte safety ceiling. Reason: extremely large tool calls "
        "can fill the context window and break JSON parsing.\n\n"
        "HOW TO RECOVER:\n"
        f"  1. Write a complete smaller concrete change for `{path_hint}`.\n"
        "  2. If the content is repetitive data, use compact code or concise "
        "source instead of embedding one huge literal.\n"
        "  3. Continue the same task and verify the result.\n\n"
        "Do not retry the exact same oversized call."
    )


def track_tool_result(
    *,
    tool_name: str,
    args: dict[str, Any],
    tool_result: str,
    round_num: int,
    state: ToolExecutionState,
    dedup_stub: str | None,
) -> None:
    failed = tool_result_is_error(tool_result)
    if failed:
        key = (tool_name, canonical_args(args))
        state.failed_calls[key] = state.failed_calls.get(key, 0) + 1

    if tool_name == "bash":
        cmd = str(args.get("command", ""))
        if failed:
            cmd_key = re.sub(r"\s+", " ", cmd.strip())
            if cmd_key:
                state.bash_failures[cmd_key] = state.bash_failures.get(cmd_key, 0) + 1
    elif tool_name == "read_file":
        read_path = args.get("path") or args.get("file_path") or ""
        if isinstance(read_path, str) and read_path:
            state.files_read[read_path] = round_num
    elif tool_name in {"list_files", "glob", "grep"}:
        if dedup_stub is None:
            state.io_calls[(tool_name, canonical_args(args))] = round_num
    elif tool_name in {"write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}:
        modified_path = args.get("path") or args.get("file_path") or ""
        if isinstance(modified_path, str) and modified_path and not failed:
            state.files_modified.add(modified_path)
            state.files_read.pop(modified_path, None)
            state.io_calls.clear()
    if (
        tool_name in {"write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}
        and not failed
    ):
        changed_path = args.get("path")
        if isinstance(changed_path, str) and changed_path and changed_path not in state.changed_files:
            state.changed_files.append(changed_path)
    # Successful repeats — feeds `repeat_stub_for_tool`.
    if tool_name in _REPEAT_STUB_TOOLS and not failed:
        key = (tool_name, canonical_args(args))
        state.success_counts[key] = state.success_counts.get(key, 0) + 1


def tool_result_is_error(tool_result: str) -> bool:
    lower = tool_result.lower()
    return (
        tool_result.startswith("Error:")
        or tool_result.startswith("error:")
        or tool_result.startswith("REJECTED")
        or tool_result.startswith("[exit code ")
        or lower.startswith("file not found:")
        or lower.startswith("directory not found:")
        or "old_string not found" in lower
        or "no-op edit" in lower
        or "applied 0/" in lower
    )
