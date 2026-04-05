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
    lines.append("To use a tool, output a JSON line: {\"tool\": \"name\", \"args\": {\"key\": \"value\"}}")
    lines.append("")
    lines.append("For write_file, put the code in a fenced block AFTER the JSON:")
    lines.append('{\"tool\": \"write_file\", \"args\": {\"path\": \"game.py\"}}')
    lines.append("```")
    lines.append("code here")
    lines.append("```")
    lines.append("")
    lines.append("IMPORTANT: Build code INCREMENTALLY. Don't write everything at once.")
    lines.append("Step 1: write_file with a basic scaffold (imports, boilerplate, main)")
    lines.append("Step 2: edit_file to add the next feature")
    lines.append("Step 3: edit_file to add the next feature")
    lines.append("This way each step is fast and the user sees progress.")
    lines.append("")
    lines.append("I will execute each tool and show you the result. Then continue.")
    lines.append("When done, respond with plain text (no JSON). One tool per response.")
    return "\n".join(lines)


def parse_tool_call(text: str) -> dict | None:
    """Try to parse a tool call from model output.

    Supports two formats:
    1. Full JSON: {"tool": "write_file", "args": {"path": "x", "content": "..."}}
    2. Hybrid: {"tool": "write_file", "args": {"path": "x"}} followed by ```code```
       (faster — model doesn't need to JSON-escape the code)

    Returns {"tool": str, "args": dict} or None if it's plain text.
    """
    text = text.strip()

    # Quick check: does it contain a tool call?
    if '{"tool"' not in text and '"tool"' not in text:
        return None

    # Try hybrid format first: JSON line + code block
    # {"tool": "write_file", "args": {"path": "game.py"}}
    # ```
    # code here
    # ```
    json_line_match = re.search(r'(\{[^{}]*"tool"\s*:\s*"(\w+)"[^{}]*\})', text)
    if json_line_match:
        try:
            tool_call = json.loads(json_line_match.group(1))
            tool_name = tool_call.get("tool", "")
            args = tool_call.get("args", {})

            # If it's write_file and content is missing, look for a code block
            if tool_name == "write_file" and "content" not in args:
                code_match = re.search(r'```\w*\n(.*?)```', text, re.DOTALL)
                if code_match:
                    args["content"] = code_match.group(1).strip()
                    tool_call["args"] = args

            return tool_call
        except json.JSONDecodeError:
            pass

    # Full JSON format
    try:
        # Find the most complete JSON object
        for m in re.finditer(r'\{[^{}]*"tool"[^{}]*\{.*?\}[^{}]*\}', text, re.DOTALL):
            parsed = json.loads(m.group())
            if isinstance(parsed, dict) and "tool" in parsed:
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Last resort: try parsing the whole text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

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
        # Skip thinking — it adds 1min+ delay on 26B with no benefit for tool calls.
        # The model plans fine without explicit thinking mode.
        use_think = False

        # Stream response — user sees tokens arriving in real time
        chunks: list[str] = []
        token_count = 0
        if round_num == 0:
            out.set_stage("thinking")
        else:
            out.set_stage(f"round {round_num + 1}")
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
                # Update indicator every few tokens so user sees activity
                if token_count % 10 == 0:
                    out.set_stage(f"generating ({token_count} tok)")
                # Also feed content to thinking peek for visibility
                preview = chunk.replace("\n", " ").strip()
                if preview and len(preview) > 3:
                    out.feed_thinking(preview)

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
