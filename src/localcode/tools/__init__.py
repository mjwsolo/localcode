"""Tool registry for the agent loop.

Each tool has its own module under this package exporting:
  * SCHEMA — the OpenAI function schema
  * execute(ctx, args) — the implementation

This module assembles the registry into two forms:
  * ALL_SCHEMAS — list of schema dicts, passed to llama-server's
    `tools` parameter so the model sees them.
  * dispatch(name, ctx, args) — name → execute resolver used by
    `src/localcode/agent.py:_execute_tool`.

Adding a tool:
  1. Create src/localcode/tools/<tool_name>.py with SCHEMA + execute.
  2. Import it in this file.
  3. Register it in `_TOOLS` below.

That's it. No edits to agent.py for the happy path (plan-mode gating
still lives there because it's cross-tool policy).
"""
from __future__ import annotations

from typing import Callable

from . import (
    agent,
    append_file,
    bash,
    background_process,
    code_navigation,
    edit_diff,
    edit_file,
    glob_tool,
    grep,
    launch_app,
    list_files,
    multi_edit,
    plan_mode,
    read_file,
    skill_tool,
    todo_write,
    web_fetch,
    web_search,
    write_file,
)
from .base import ToolContext, ToolResult
from .facts import extract_tool_facts

# name → (schema, executor). For plan_mode the module provides two
# (schema, executor) pairs under a single file.
_TOOLS: dict[str, tuple[dict, Callable[[ToolContext, dict], str]]] = {
    "read_file":      (read_file.SCHEMA,      read_file.execute),
    "write_file":     (write_file.SCHEMA,     write_file.execute),
    "append_file":    (append_file.SCHEMA,    append_file.execute),
    "edit_file":      (edit_file.SCHEMA,      edit_file.execute),
    "multi_edit":     (multi_edit.SCHEMA,     multi_edit.execute),
    "edit_diff":      (edit_diff.SCHEMA,      edit_diff.execute),
    "bash":           (bash.SCHEMA,           bash.execute),
    "background_process": (background_process.SCHEMA, background_process.execute),
    "code_navigation": (code_navigation.SCHEMA, code_navigation.execute),
    "launch_app":     (launch_app.SCHEMA,     launch_app.execute),
    "grep":           (grep.SCHEMA,           grep.execute),
    "glob":           (glob_tool.SCHEMA,      glob_tool.execute),
    "list_files":     (list_files.SCHEMA,     list_files.execute),
    "web_search":     (web_search.SCHEMA,     web_search.execute),
    "web_fetch":      (web_fetch.SCHEMA,      web_fetch.execute),
    "skill":          (skill_tool.SCHEMA,     skill_tool.execute),
    "todo_write":     (todo_write.SCHEMA,     todo_write.execute),
    "agent":          (agent.SCHEMA,          agent.execute),
    "enter_plan_mode": (plan_mode.ENTER_SCHEMA, plan_mode.execute_enter),
    "exit_plan_mode":  (plan_mode.EXIT_SCHEMA,  plan_mode.execute_exit),
}

_PUBLIC_TOOL_NAMES = [
    "read_file",
    "write_file",
    "append_file",
    "edit_file",
    "multi_edit",
    "edit_diff",
    "bash",
    "background_process",
    "code_navigation",
    "launch_app",
    "grep",
    "glob",
    "list_files",
    "web_search",
    "web_fetch",
    "skill",
    "todo_write",
    "agent",
]

ALL_SCHEMAS: list[dict] = [
    _TOOLS[name][0]
    for name in _PUBLIC_TOOL_NAMES
    if name in _TOOLS
]

AVAILABLE_NAMES: list[str] = list(_PUBLIC_TOOL_NAMES)

def schemas_for_names(
    names: list[str] | tuple[str, ...] | set[str],
    *,
    content_max_chars: int | None = None,
) -> list[dict]:
    """Return schemas in registry order for a selected tool subset."""
    wanted = {str(name).strip() for name in names}
    return [
        _TOOLS[name][0]
        for name in _PUBLIC_TOOL_NAMES
        if name in wanted and name in _TOOLS
    ]


def schemas_for_goal(
    goal_type: str,
    user_text: str = "",
    task_stage: str = "",
    recovery_mode: str = "",
) -> list[dict]:
    """Return the flat coding tool surface used every round.

    Tool routing, build stages, and recovery-specific schema mutation made the
    controller brittle. The model sees the same compact set every round; tools
    return actionable errors and the loop may add at most two generic
    correction nudges.
    """
    selected = {
        "read_file", "bash", "edit_file", "write_file", "append_file", "list_files",
        "background_process", "code_navigation",
        # Search/discovery — without these the model resorts to bash+grep / curl
        # for everything and the user sees the "google scraped HTML" failure mode.
        "grep", "glob", "web_search", "web_fetch", "skill",
        # Sub-agent — lets the model spawn focused sub-tasks (explore/plan/verify
        # /general-purpose) so it doesn't burn its own context on long searches.
        "agent",
        # Working-memory checklist — lets the model record a plan and track
        # done/in-progress/remaining across rounds so it stops repeating work.
        "todo_write",
    }
    schemas = schemas_for_names(selected)
    # MCP tools — any tools exposed by user-configured MCP servers in
    # ~/.localcode/mcp.json. Names are prefixed `mcp_<server>_<tool>`
    # so they don't collide with built-ins. Best-effort: if no servers
    # are configured or any fail to connect, we just return built-ins.
    try:
        from ..mcp import mcp_tool_schemas
        schemas = schemas + mcp_tool_schemas()
    except Exception:
        pass
    return schemas

_MODULES = {
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "edit_file": edit_file,
    "multi_edit": multi_edit,
    "edit_diff": edit_diff,
    "bash": bash,
    "launch_app": launch_app,
    "grep": grep,
    "glob": glob_tool,
    "list_files": list_files,
    "web_search": web_search,
    "web_fetch": web_fetch,
    "skill": skill_tool,
    "todo_write": todo_write,
}


def is_concurrency_safe(name: str, args: dict) -> bool:
    mod = _MODULES.get((name or "").strip())
    checker = getattr(mod, "is_concurrency_safe", None)
    if checker is None:
        return False
    try:
        return bool(checker(args))
    except Exception:
        return False


def _format_missing_args_error(tool_name: str, schema: dict, args: dict,
                               missing: list[str]) -> str:
    """Build a model-actionable error when required args are missing.

    The old message — "Error: 'path' argument is required for write_file."
    — told the model WHICH field was missing but not WHAT IT DID emit,
    WHY it's wrong, or HOW to fix it. IQ3 quant would read that, think
    about it, retry with the SAME missing arg, get the same error, and
    eventually spiral into a "I'll use bash / no, write_file / no,
    bash" repetition loop (the failure mode in images 103-105).

    This message echoes the exact args the model emitted, spells out
    what's missing with the schema, and shows the correct call shape
    inline — giving the model everything it needs to self-correct on
    the very next round instead of guessing."""
    import json as _json
    required = schema.get("function", {}).get("parameters", {}).get("required", [])

    # Short snippet of what the model DID send, so it can compare.
    try:
        got_preview = _json.dumps(
            {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v)
             for k, v in args.items()},
            ensure_ascii=False,
        )
    except Exception:
        got_preview = str(args)[:200]

    example_args = {}
    for field in required:
        if field in args:
            example_args[field] = args[field]
        else:
            # Placeholder spelling out what the model needs to provide.
            example_args[field] = f"<{field} value — REQUIRED>"
    try:
        example_call = _json.dumps(example_args, ensure_ascii=False)
    except Exception:
        example_call = str(example_args)

    return (
        f"Error: {tool_name} was called without required argument(s): "
        f"{', '.join(repr(m) for m in missing)}.\n"
        f"You sent: {got_preview}\n"
        f"Required fields for {tool_name}: {required}.\n"
        f"Retry by calling {tool_name} with ALL required fields filled in, "
        f"e.g. arguments={example_call}"
    )


def dispatch_result(name: str, ctx: ToolContext, args: dict) -> ToolResult:
    # Strip whitespace before lookup. Quantized models (notably Qwen 3.6
    # IQ2_M) sometimes emit tool names with a trailing or leading space
    # — `'list_files '` instead of `'list_files'` — which would cause an
    # otherwise-valid call to fail.
    clean = (name or "").strip()
    # MCP tool? Dispatch to the right server and return its text content.
    # These tool names have the form mcp_<server>_<tool> and don't appear
    # in _TOOLS — they're added dynamically by mcp_tool_schemas().
    if clean.startswith("mcp_"):
        try:
            from ..mcp import dispatch_mcp_tool
            text = dispatch_mcp_tool(clean, args if isinstance(args, dict) else {})
            if text is not None:
                ok = not str(text).startswith("REJECTED:") and not str(text).startswith("MCP call ")
                return ToolResult(text=str(text), ok=ok,
                                  facts={"tool": clean, "ok": ok})
        except Exception as e:
            return ToolResult(text=f"MCP dispatch error: {e}", ok=False,
                              facts={"tool": clean, "ok": False})
    pair = _TOOLS.get(clean)
    if pair is None:
        # Use the error-code system so the user gets a documented [Eccc]
        # prefix instead of a free-text string. Code E2101 is the
        # "Unknown tool" entry; see src/localcode/errors.py.
        from ..errors import LocalCodeError, by_code
        code = by_code("E2101")
        if code is not None:
            text = str(LocalCodeError(code=code,
                                      detail=f"'{name}'. Available: " + ", ".join(AVAILABLE_NAMES)))
            return ToolResult(text=text, ok=False, facts={"tool": clean, "ok": False})
        # Fallback (registry missing — shouldn't happen in production)
        text = f"[E2101] Unknown tool: '{name}'. Available: " + ", ".join(AVAILABLE_NAMES)
        return ToolResult(text=text, ok=False, facts={"tool": clean, "ok": False})
    schema, executor = pair

    # Pre-dispatch schema validation. Without this, each tool did its
    # own "is field X present?" check and returned a terse error that
    # failed to tell the model what it *did* emit. Centralising here
    # means every tool benefits — and the message can reference the
    # schema's `required` list concretely. See
    # `_format_missing_args_error` for the root-cause rationale.
    if isinstance(args, dict):
        required = schema.get("function", {}).get("parameters", {}).get("required", []) or []
        missing = [f for f in required if f not in args or args[f] is None]
        if missing:
            text = _format_missing_args_error(clean, schema, args, missing)
            return ToolResult(text=text, ok=False, facts={"tool": clean, "ok": False, "missing_args": missing})
    else:
        # Model emitted something non-dict for args. Recoverable by the
        # model if told clearly what we expected.
        text = (
            f"Error: {clean} expected a JSON object for arguments, got {type(args).__name__}. "
            f"Retry with a JSON object like {{\"field\": \"value\"}}."
        )
        return ToolResult(text=text, ok=False, facts={"tool": clean, "ok": False})

    # Tool-call telemetry — one append-only line per dispatch in the
    # project's lifecycle.log. Lets the user (and us) answer "which
    # tools is the model actually picking?" without diffing SQLite
    # session stores. Records the tool name + a short args summary
    # (path / command first 60 chars, never full content). Best-effort:
    # any failure here is silently swallowed so the tool itself still
    # runs and returns its real result.
    try:
        from ..server_manager import _lifecycle_log
        if clean == "edit_file":
            # Capture short previews of old/new so log analysis can spot
            # edit oscillation (model edits A→B then B→A repeatedly) and
            # near-no-op patterns. Truncated to 60 chars each so the
            # event stays small but the first non-whitespace token is
            # always visible. Real failure 2026-04-26: 12+ edit_file
            # calls returning "1 replacement" each — we couldn't tell
            # from logs whether they were the same edit or a flip-flop
            # because old_string/new_string weren't logged.
            _o = (args.get("old_string", "") or "").strip().replace("\n", "↵")[:60]
            _n = (args.get("new_string", "") or "").strip().replace("\n", "↵")[:60]
            arg_summary = (
                f"path={args.get('path', '?')!r} "
                f"old={_o!r} new={_n!r}"
            )
        elif clean in ("write_file", "append_file", "multi_edit"):
            arg_summary = f"path={args.get('path', '?')!r}"
        elif clean == "bash":
            arg_summary = f"cmd={(args.get('command', '') or '')[:60]!r}"
        elif clean in ("read_file", "list_files", "glob"):
            arg_summary = f"path={args.get('path', args.get('pattern', '?'))!r}"
        elif clean == "grep":
            arg_summary = f"pat={args.get('pattern', '?')!r}"
        else:
            arg_summary = ""
        _lifecycle_log("tool_call", name=clean, **(
            {"args": arg_summary} if arg_summary else {}
        ))
    except Exception:
        pass

    raw_result = executor(ctx, args)
    result = raw_result if isinstance(raw_result, ToolResult) else _normalize_result(clean, args, str(raw_result))

    return result


def dispatch(name: str, ctx: ToolContext, args: dict) -> str:
    return dispatch_result(name, ctx, args).text


def _normalize_result(tool_name: str, args: dict, text: str) -> ToolResult:
    facts = extract_tool_facts(tool_name, args, text)
    return ToolResult(text=text, ok=bool(facts.get("ok", True)), facts=facts)


__all__ = [
    "ALL_SCHEMAS",
    "AVAILABLE_NAMES",
    "ToolContext",
    "ToolResult",
    "dispatch",
    "dispatch_result",
    "schemas_for_goal",
    "schemas_for_names",
]
