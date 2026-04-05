"""Text-based tool calling loop — the model drives tools via JSON, not special tokens.

Why: Quantized local models (IQ3_S, Q4) can't reliably produce native tool call tokens.
But they CAN output valid JSON consistently. So we describe tools in the system prompt
and parse JSON tool calls from the model's text output.

This is how Aider, and most local-first tools work. The model drives itself — just like
Codex — but using text JSON instead of native function calling tokens.

Flow:
  1. System prompt describes available tools as text
  2. Model outputs {"tool": "name", "args": {...}} to call a tool
  3. We parse, execute, and feed the result back
  4. Model continues until it responds with plain text (no tool call)
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import GemApp
    from .output import OutputManager


# Tools exposed to the model via text (not Ollama native)
TEXT_TOOLS = {
    "write_file": {
        "desc": "Create or overwrite a file",
        "args": "path (string), content (string)",
    },
    "edit_file": {
        "desc": "Edit a file by replacing text",
        "args": "path (string), old_string (string), new_string (string)",
    },
    "read_file": {
        "desc": "Read a file's contents",
        "args": "path (string)",
    },
    "bash": {
        "desc": "Run a shell command",
        "args": "command (string)",
    },
    "grep": {
        "desc": "Search file contents with regex",
        "args": "pattern (string), path (string, optional)",
    },
    "glob": {
        "desc": "Find files matching a pattern",
        "args": "pattern (string)",
    },
    "web_search": {
        "desc": "Search the web",
        "args": "query (string)",
    },
    "current_datetime": {
        "desc": "Get current date and time",
        "args": "(none)",
    },
}


def build_tool_system_prompt(tools: dict[str, dict] | None = None) -> str:
    """Build the system prompt section describing available tools."""
    tools = tools or TEXT_TOOLS
    lines = ["You have these tools:"]
    for name, info in tools.items():
        lines.append(f"  {name}({info['args']}): {info['desc']}")
    lines.append("")
    lines.append('To use a tool, output JSON: {"tool": "name", "args": {"key": "value"}}')
    lines.append("I will execute it and show you the result. Then continue.")
    lines.append("When the task is fully done, respond with plain text (no JSON).")
    lines.append("You can call multiple tools — one at a time. I'll show each result.")
    return "\n".join(lines)


def parse_tool_call(text: str) -> dict | None:
    """Try to parse a JSON tool call from model output.

    Returns {"tool": str, "args": dict} or None if it's plain text.
    """
    text = text.strip()

    # Quick check: does it look like JSON?
    if not (text.startswith("{") or '{"tool"' in text):
        return None

    # Try to extract JSON object
    match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{.*?\}\s*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing the whole text as JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find JSON embedded in text (model might add explanation around it)
    for m in re.finditer(r'\{.*?\}', text, re.DOTALL):
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict) and "tool" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def execute_tool(app: "GemApp", tool_name: str, args: dict) -> str:
    """Execute a tool call and return the result as text."""
    try:
        if tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            if not path or not content:
                return "Error: write_file needs path and content"
            full_path = app.repo_root / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            app.toolkit.changes.snapshot_before(path, "text_tool")
            full_path.write_text(content)
            lines = len(content.splitlines())
            return f"Written {path} ({lines} lines)"

        elif tool_name == "edit_file":
            path = args.get("path", "")
            old = args.get("old_string", "")
            new = args.get("new_string", "")
            if not path:
                return "Error: edit_file needs path"
            full_path = app.repo_root / path
            if not full_path.is_file():
                return f"Error: {path} not found"
            text = full_path.read_text(errors="replace")
            if old not in text:
                return f"Error: old_string not found in {path}"
            app.toolkit.changes.snapshot_before(path, "text_tool_edit")
            text = text.replace(old, new, 1)
            full_path.write_text(text)
            return f"Edited {path}"

        elif tool_name == "read_file":
            path = args.get("path", "")
            full_path = app.repo_root / path
            if not full_path.is_file():
                return f"Error: {path} not found"
            content = full_path.read_text(errors="replace")
            if len(content) > 8000:
                content = content[:8000] + "\n... (truncated)"
            return content

        elif tool_name == "bash":
            import subprocess
            cmd = args.get("command", "")
            if not cmd:
                return "Error: bash needs command"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=str(app.repo_root),
            )
            output = (result.stdout + result.stderr).strip()
            if len(output) > 4000:
                output = output[:4000] + "\n... (truncated)"
            return output or "(no output)"

        elif tool_name == "grep":
            import subprocess
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
                 "--include=*.json", "--include=*.md", pattern, path],
                capture_output=True, text=True, timeout=10,
                cwd=str(app.repo_root),
            )
            output = result.stdout.strip()
            if len(output) > 4000:
                output = output[:4000] + "\n... (truncated)"
            return output or "No matches"

        elif tool_name == "glob":
            import subprocess
            pattern = args.get("pattern", "**/*")
            result = subprocess.run(
                ["find", ".", "-name", pattern, "-not", "-path", "*/.git/*",
                 "-not", "-path", "*/__pycache__/*"],
                capture_output=True, text=True, timeout=10,
                cwd=str(app.repo_root),
            )
            return result.stdout.strip() or "No files found"

        elif tool_name == "web_search":
            query = args.get("query", "")
            # Use existing web search tool
            call = {"function": {"name": "web_search", "arguments": {"query": query}}}
            results = app.toolkit.execute_tool_calls([call])
            return results[0].get("content", "No results") if results else "No results"

        elif tool_name == "current_datetime":
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

        else:
            return f"Unknown tool: {tool_name}"

    except Exception as exc:
        return f"Error: {exc}"


def run_text_tool_loop(
    app: "GemApp",
    user_text: str,
    composed_messages: list[dict],
    out: "OutputManager",
    max_rounds: int = 15,
) -> str:
    """Main loop: model calls tools via JSON text, we execute, model continues.

    This replaces the native Ollama tool calling loop for models that
    can't reliably produce special tool tokens (quantized models).
    """
    # Build system prompt with tool descriptions
    tool_prompt = build_tool_system_prompt()

    # Inject tool descriptions into the system message
    messages = list(composed_messages)
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + "\n\n" + tool_prompt,
        }
    else:
        messages.insert(0, {"role": "system", "content": tool_prompt})

    final_text = ""

    for round_num in range(max_rounds):
        # Think on first round (the model's "planning" phase)
        use_think = round_num == 0

        # Stream the response so user sees progress (not blank for 1min)
        chunks: list[str] = []
        token_count = 0
        for event in app.engine.stream_chat_events(messages, think=use_think):
            if event["type"] == "thinking":
                # Show thinking peek — user sees the model planning
                chunk = str(event["content"])
                peek = chunk.replace("\n", " ").strip()
                if peek and len(peek) > 3:
                    out.feed_thinking(peek)
            elif event["type"] == "content":
                chunk = str(event["content"])
                # Filter special tokens
                if "<|" in chunk or "|>" in chunk:
                    import re as _re
                    chunk = _re.sub(r'<\|[^>]*\|>', '', chunk)
                chunks.append(chunk)
                token_count += 1
                # Update indicator with token count so user sees progress
                if token_count % 20 == 0:
                    out.set_stage(f"generating ({token_count} tokens)")

        content = "".join(chunks).strip()

        if not content:
            break

        # Try to parse as a tool call
        tool_call = parse_tool_call(content)

        if tool_call:
            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})

            # Show what's happening
            args_preview = str(tool_args.get("path", tool_args.get("command", tool_args.get("query", ""))))[:60]
            out.log_tool(tool_name, args_preview)

            # Permission check
            allowed, reason = app.perms.check(tool_name, tool_args)
            if not allowed:
                result = f"Denied: {reason}"
                out.tool_result(result, error=True)
            else:
                result = execute_tool(app, tool_name, tool_args)
                is_err = result.startswith("Error")
                out.tool_result(result[:120], error=is_err)

            # Feed result back to model
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Tool result:\n{result}\n\nContinue. Call another tool or give your final answer."})
            # Restart indicator for next round
            out.start_thinking()

        else:
            # Plain text response — model is done, stream it to user
            final_text = content
            out.stream(content)
            break

    return final_text
