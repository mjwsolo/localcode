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
    lines.append("IMPORTANT:")
    lines.append("- For NEW files: write_file with complete code.")
    lines.append("- For CHANGES to existing files: use edit_file with small, surgical old_string/new_string.")
    lines.append("  Do NOT rewrite the entire file. Only change the lines that need changing.")
    lines.append("- MINIMAL changes. If the fix is 2 lines, change 2 lines. Not 200.")
    lines.append("")
    lines.append("I will execute each tool and show you the result. Then continue.")
    lines.append("When done, respond with plain text (no JSON). One tool per response.")
    lines.append("NEVER run GUI apps (pygame, tkinter) via bash — they block forever. Just tell the user how to run it.")
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

    # Strategy 1: Look for a code block after a JSON tool call line
    # {"tool": "write_file", "args": {"path": "game.py"}}
    # ```python
    # code here
    # ```
    code_block_match = re.search(r'```\w*\n(.*?)```', text, re.DOTALL)
    # Match JSON with one level of nested braces: {"tool": "x", "args": {"key": "val"}}
    simple_json = re.search(r'(\{"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\})', text)
    if simple_json and code_block_match:
        try:
            tool_call = json.loads(simple_json.group(1))
            if tool_call.get("tool") == "write_file":
                tool_call.setdefault("args", {})["content"] = code_block_match.group(1).strip()
                return tool_call
        except json.JSONDecodeError:
            pass

    # Strategy 2: Full JSON with nested args (model put content in JSON)
    # Find the largest valid JSON object containing "tool"
    # Use a greedy approach since content may have special chars
    start = text.find('{"tool"')
    if start == -1:
        start = text.find('"tool"')
    if start >= 0:
        # Try parsing from this point with increasing length
        substr = text[start:]
        # Try the whole thing first
        for end_pos in range(len(substr), 10, -1):
            candidate = substr[:end_pos]
            if not candidate.rstrip().endswith("}"):
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "tool" in parsed:
                    # Unescape content if it has literal \n
                    args = parsed.get("args", {})
                    if "content" in args and isinstance(args["content"], str):
                        if "\\n" in args["content"]:
                            args["content"] = args["content"].replace("\\n", "\n").replace("\\t", "\t")
                            parsed["args"] = args
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
                timeout=120, cwd=str(app.repo_root),
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
    """Simple tool loop: model generates → we parse → execute → repeat.

    No streaming tricks, no early-stop, no partial parsing.
    Just clean rounds. The indicator shows activity during the wait.
    """
    tool_prompt = build_tool_system_prompt()

    messages = list(composed_messages)
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = messages[0]["content"] + "\n\n" + tool_prompt
    else:
        messages.insert(0, {"role": "system", "content": tool_prompt})

    final_text = ""

    for round_num in range(max_rounds):
        out.set_stage(f"▶ round {round_num + 1}" if round_num > 0 else "▶ processing prompt")

        # Print the current phase as a permanent sub-item
        if round_num == 0:
            out.print_info("▶ processing prompt")
        else:
            out.print_info(f"▶ round {round_num + 1}")

        # Stream to collect full response — shows token count in indicator
        chunks: list[str] = []
        generating_printed = False
        for event in app.engine.stream_chat_events(messages, think=False):
            if event["type"] == "content":
                chunk = str(event["content"])
                if "<|" in chunk or "|>" in chunk:
                    chunk = re.sub(r'<\|[^>]*\|>', '', chunk)
                if chunk:
                    chunks.append(chunk)
                    out.feed_thinking(chunk)
                    if len(chunks) == 5 and not generating_printed:
                        # Print "generating" as a permanent sub-item once tokens flow
                        out.print_info("▶ generating")
                        generating_printed = True
                    if len(chunks) % 20 == 0:
                        out.set_stage(f"generating ({len(chunks)} tok)")

        content = "".join(chunks).strip()
        if not content:
            break

        # Try to parse a tool call
        tool_call = parse_tool_call(content)

        if tool_call:
            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})

            # Fix write_file content: unescape, extract from code block
            if tool_name == "write_file":
                file_content = tool_args.get("content", "")
                if not file_content:
                    # Try code block
                    code_match = re.search(r'```\w*\n(.*?)```', content, re.DOTALL)
                    if code_match:
                        file_content = code_match.group(1).strip()
                        tool_args["content"] = file_content
                if "\\n" in file_content:
                    tool_args["content"] = file_content.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')

            # Show and execute — appears immediately in the UI
            preview = tool_args.get("path", tool_args.get("command", tool_args.get("query", "")))
            out.log_tool(tool_name, str(preview)[:60])  # prints ● tool_name immediately
            result = execute_tool(app, tool_name, tool_args)
            is_err = result.startswith("Error")
            out.tool_result(result[:120], error=is_err)

            # Feed result back — only the tool call, not closing text
            messages.append({"role": "assistant", "content": json.dumps(tool_call)})
            messages.append({"role": "user", "content": (
                f"Tool result: {result}\n\n"
                f"Next step? Install deps, fix bugs, or say DONE if complete."
            )})
            out.start_thinking(reset=False)

        else:
            # No tool call — check for code block we should write
            code_match = re.search(r'```\w*\n(.*?)```', content, re.DOTALL)
            path_hint = re.search(r'`(\w+\.(?:py|js|ts|html|css|json|md|txt))`', content)
            if code_match and path_hint:
                out.log_tool("write_file", path_hint.group(1))
                result = execute_tool(app, "write_file", {
                    "path": path_hint.group(1),
                    "content": code_match.group(1).strip(),
                })
                out.tool_result(result[:120])

            # Stream the text to user
            final_text = content.split("```")[0].strip() if code_match else content
            out.stream(final_text)
            break

    return final_text
