"""agent — spawn a sub-agent to handle a focused multi-step task.

A sub-agent is a fresh model session with:
  - its own system prompt (specialized for the agent type)
  - a tool subset (so e.g. an "explore" agent only gets read tools)
  - the same model + runtime as the parent
  - independent message history (results bubble back to the parent
    only via the final assistant message)

Built-in agent types (registry below) mirror Codex's pattern:
  - general-purpose  : broad research / multi-step tasks (all tools)
  - explore          : codebase search + read only (no edits)
  - plan             : reads + outputs structured plan, no edits
  - verify           : runs tests/builds, no edits

Calling pattern from the parent model:
  {"name": "agent", "input": {
      "subagent_type": "explore",
      "description": "find all CSV parsing call-sites",
      "prompt": "search for csv.reader and pandas.read_csv usage..."
  }}

Returns the sub-agent's final assistant text as the tool result.
"""
from __future__ import annotations

from .base import ToolContext


SCHEMA = {
    "type": "function",
    "function": {
        "name": "agent",
        "description": (
            "Spawn a sub-agent to handle a focused, multi-step task. Use for "
            "open-ended research (codebase searches, design exploration) that "
            "would otherwise consume many tool calls in your own context. "
            "Each subagent_type has a curated tool subset and system prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subagent_type": {
                    "type": "string",
                    "enum": ["general-purpose", "explore", "plan", "verify"],
                    "description": (
                        "Which built-in sub-agent to spawn. "
                        "'general-purpose' has all tools; 'explore' is read-only "
                        "(grep/glob/read for codebase mapping); 'plan' is read-only "
                        "and returns a structured implementation plan; 'verify' "
                        "runs tests and bash for verification."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Short (3-5 word) label for what the agent will do — "
                        "shown to the user in the activity feed."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The full task for the sub-agent. Self-contained — the "
                        "sub-agent does not see your context. Include all "
                        "background, constraints, and the exact deliverable."
                    ),
                },
            },
            "required": ["subagent_type", "description", "prompt"],
        },
    },
}


# Built-in sub-agent registry. Each entry defines the tool subset + a
# specialized system prompt. Adding new agents is one dict entry; no
# changes to the dispatch path needed.
_AGENTS = {
    "general-purpose": {
        "tools": [
            "read_file", "write_file", "edit_file", "append_file",
            "bash", "grep", "glob", "list_files", "web_search", "web_fetch",
        ],
        "system": (
            "You are a sub-agent spawned by the main LocalCode agent. Your "
            "scope is the task the parent gave you. Complete it fully, then "
            "respond with a concise report — what you did, what you found, "
            "and any blockers. The caller will relay this verbatim to the "
            "user. Stay tightly focused; don't gold-plate."
        ),
    },
    "explore": {
        "tools": ["read_file", "grep", "glob", "list_files"],
        "system": (
            "You are an exploration sub-agent. READ-ONLY — you cannot edit "
            "or write files. Use grep/glob/read/list_files to map the "
            "codebase the parent agent is asking about. Report file paths "
            "with line numbers and short excerpts. Optimize for breadth "
            "and accuracy, not for ideas."
        ),
    },
    "plan": {
        "tools": ["read_file", "grep", "glob", "list_files"],
        "system": (
            "You are a planning sub-agent. READ-ONLY. Read the relevant "
            "code, then return a numbered implementation plan: file paths, "
            "function names, what to change. Identify risks. Do NOT write "
            "any code or files."
        ),
    },
    "verify": {
        "tools": ["read_file", "bash", "grep", "glob", "list_files"],
        "system": (
            "You are a verification sub-agent. Run the tests, builds, or "
            "checks the parent requested. Read the relevant code to ground "
            "your verification. Do NOT make edits — only run and report."
        ),
    },
}


def _dispatch_guarded(ctx: ToolContext, name: str, tool_args: dict) -> str:
    """Run one sub-agent tool call through the SAME guarded path the parent uses.

    Calling the tool's raw executor here would bypass every protection the
    main loop applies: the autonomy-independent `_safety_hard_block`, the
    pre_tool_use hook veto, the destructive-write guards, and the user
    approval gate. A sub-agent is still the same process acting on the same
    machine, so it gets the same gate — one `agent` call must not buy
    unapproved shell/file access.

    Imports are function-local: `localcode.agent.helpers` imports
    `localcode.tools`, so a module-level import here would be circular.
    """
    from ..agent.helpers import (
        _execute_tool_result,
        _first_token,
        _needs_confirmation,
        _request_approval_verdict,
    )

    app = getattr(ctx, "app", None)
    out = getattr(ctx, "out", None)

    # Approval gate — identical semantics to agent/loop.py: verdict is one of
    # "once" / "always" / "deny"; "always" whitelists the command's first
    # token for the session; "deny" refuses the call.
    try:
        needs = _needs_confirmation(name, tool_args, app)
    except Exception:
        needs = True  # fail closed
    if needs:
        cmd = tool_args.get("command", "")
        if not cmd:
            _wpath = tool_args.get("path") or tool_args.get("file_path") or ""
            cmd = f"{name} {_wpath}".strip()
        try:
            verdict = _request_approval_verdict(app, out, name, cmd)
        except Exception:
            verdict = "deny"
        if verdict == "always":
            allow_set = getattr(app, "_session_allow", None)
            if allow_set is None and app is not None:
                app._session_allow = set()
                allow_set = app._session_allow
            if allow_set is not None:
                allow_set.add(_first_token(cmd))
        elif verdict != "once":
            return "Denied by user."

    try:
        return _execute_tool_result(app, name, tool_args, out).text
    except Exception as e:
        return f"Tool {name} raised: {e}"


def execute(ctx: ToolContext, args: dict) -> str:
    """Run a sub-agent loop with the requested type + prompt.

    Implementation: reuses the parent's `LocalCodeApp` for the runtime
    gateway (same llama-server, same model) but constructs a fresh
    message list with the sub-agent's system prompt and only the tool
    subset the agent is permitted to use. The loop runs synchronously
    in the caller's thread — the parent's bash tool is also synchronous,
    so this matches existing latency expectations.
    """
    subagent_type = args.get("subagent_type", "general-purpose")
    description = args.get("description", "sub-agent task")
    prompt = args.get("prompt", "").strip()
    if not prompt:
        return "REJECTED: agent prompt is required."
    spec = _AGENTS.get(subagent_type)
    if spec is None:
        return f"REJECTED: unknown subagent_type {subagent_type!r}. " \
               f"Valid: {list(_AGENTS.keys())}"

    # Resolve tool subset
    try:
        from . import schemas_for_names, _TOOLS
    except ImportError:
        return "REJECTED: tool registry unavailable."
    tool_schemas = schemas_for_names(spec["tools"])
    tool_map = {n: _TOOLS[n] for n in spec["tools"] if n in _TOOLS}

    # Build the message list. The sub-agent's system prompt is fully
    # self-contained — it does NOT inherit the parent's prompt.
    messages = [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": prompt},
    ]

    # Reuse the parent app's runtime gateway. Stream tokens, dispatch
    # tool calls, accumulate final assistant text. Hard cap on rounds
    # so a runaway sub-agent can't burn the parent's budget.
    MAX_ROUNDS = 12
    final_text_parts: list[str] = []
    try:
        engine = ctx.app.engine  # LocalCodeApp -> .engine = LocalCodeRuntimeGateway
    except AttributeError:
        return "REJECTED: parent backend not initialized."

    for _round in range(MAX_ROUNDS):
        # One model turn
        try:
            response = engine.chat(messages, tools=tool_schemas, stream=False)
        except Exception as e:
            return f"Sub-agent error: {e}"
        assistant_text = response.get("content", "") or ""
        tool_calls = response.get("tool_calls", []) or []
        if assistant_text:
            final_text_parts.append(assistant_text)
        if not tool_calls:
            break  # natural completion
        # Dispatch each tool call and append the result as a tool message
        messages.append({"role": "assistant",
                         "content": assistant_text,
                         "tool_calls": tool_calls})
        for call in tool_calls:
            name = call.get("function", {}).get("name", "")
            raw_args = call.get("function", {}).get("arguments", "{}")
            try:
                import json as _json
                tool_args = _json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                tool_args = {}
            entry = tool_map.get(name)
            if entry is None:
                tool_result = f"Tool {name!r} not available to this sub-agent."
            else:
                tool_result = _dispatch_guarded(ctx, name, tool_args)
            messages.append({"role": "tool",
                             "tool_call_id": call.get("id", ""),
                             "content": str(tool_result)[:8000]})
    return ("\n\n".join(final_text_parts).strip()
            or f"Sub-agent ({subagent_type}) completed without output.")


def is_concurrency_safe(args: dict) -> bool:
    # Sub-agents may invoke bash/edit; not safe to run two at once.
    return False
