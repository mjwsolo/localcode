"""Agent tool loop — modeled after OpenAI Codex CLI architecture.

The loop is simple:
  1. Send messages to model
  2. Model responds with text
  3. If text contains a tool call → parse, execute, append result, loop
  4. If text is plain (no tool call) → model is done, show to user

The model decides when to stop by simply NOT making a tool call.
No "done" signal, no structured schema — just natural conversation.

Tool calls use JSON: {"tool": "name", "args": {...}}
Parsed from the model's text output (works with any model, quantized or not).
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import GemApp
    from .output import OutputManager


TOOL_PROMPT = """# Tools

You can use tools by outputting a JSON object on its own line:
{"tool": "tool_name", "args": {"param": "value"}}

After each tool call, I'll show you the result. Then keep going until the task is fully done.
When you're finished, respond with a normal text message (no JSON) summarizing what you did.

Available tools:

**write_file** — Create or overwrite a file.
  Args: path (string), content (string — use \\n for newlines)
  Example: {"tool": "write_file", "args": {"path": "game.py", "content": "import pygame\\nprint('hello')"}}

**edit_file** — Replace specific text in an existing file. Read the file first!
  Args: path (string), old_string (string), new_string (string)
  old_string must match EXACTLY. Use 2-4 lines for uniqueness.

**read_file** — Read a file's contents. Always read before editing.
  Args: path (string)

**bash** — Run a shell command.
  Args: command (string)

**grep** — Search file contents with regex.
  Args: pattern (string), path (string, optional)

**glob** — Find files by pattern.
  Args: pattern (string)

**web_search** — Search the web.
  Args: query (string)

**current_datetime** — Get current date/time.
  Args: (none)

# Guidelines

- Complete the task fully. Don't gold-plate, but don't leave it half-done.
- Make MINIMAL changes. Don't refactor or clean up code you weren't asked to change.
- Read files before editing them.
- One tool call per response.
- Keep going until done — install dependencies, fix bugs, verify if possible."""


def parse_tool_call(text: str) -> dict | None:
    """Extract a JSON tool call from model text. Returns None if no tool call found."""
    text = text.strip()

    # Quick check
    if '"tool"' not in text:
        return None

    # Find JSON objects containing "tool"
    # Try each { ... } block
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i + 1]
                if '"tool"' in candidate:
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "tool" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                start = -1

    return None


def execute_tool(app: "GemApp", tool_name: str, args: dict) -> str:
    """Execute a tool and return the result as text."""
    try:
        if tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            if not path:
                return "Error: need path"
            if not content:
                return "Error: need content"
            # Unescape JSON string escapes
            if "\\n" in content:
                content = content.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
            full = app.repo_root / path
            full.parent.mkdir(parents=True, exist_ok=True)
            app.toolkit.changes.snapshot_before(path, "agent")
            full.write_text(content)
            return f"Written {path} ({len(content.splitlines())} lines)"

        elif tool_name == "edit_file":
            path = args.get("path", "")
            old = args.get("old_string", "")
            new = args.get("new_string", "")
            if not path or not old:
                return "Error: need path and old_string"
            full = app.repo_root / path
            if not full.is_file():
                return f"Error: {path} not found"
            text = full.read_text(errors="replace")
            if old not in text:
                return f"Error: old_string not found in {path}"
            app.toolkit.changes.snapshot_before(path, "agent_edit")
            full.write_text(text.replace(old, new, 1))
            return f"Edited {path}"

        elif tool_name == "read_file":
            path = args.get("path", "")
            full = app.repo_root / path
            if not full.is_file():
                return f"Error: {path} not found"
            content = full.read_text(errors="replace")
            if len(content) > 8000:
                content = content[:8000] + "\n... (truncated)"
            return content

        elif tool_name == "bash":
            cmd = args.get("command", "")
            if not cmd:
                return "Error: need command"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=120, cwd=str(app.repo_root),
            )
            output = (result.stdout + result.stderr).strip()
            if len(output) > 4000:
                output = output[:4000] + "\n... (truncated)"
            return output or "(no output)"

        elif tool_name == "grep":
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
                 "--include=*.json", "--include=*.md", pattern, path],
                capture_output=True, text=True, timeout=10, cwd=str(app.repo_root),
            )
            return result.stdout.strip()[:4000] or "No matches"

        elif tool_name == "glob":
            pattern = args.get("pattern", "**/*")
            result = subprocess.run(
                ["find", ".", "-name", pattern, "-not", "-path", "*/.git/*",
                 "-not", "-path", "*/__pycache__/*"],
                capture_output=True, text=True, timeout=10, cwd=str(app.repo_root),
            )
            return result.stdout.strip()[:4000] or "No files found"

        elif tool_name == "web_search":
            query = args.get("query", "")
            call = {"function": {"name": "web_search", "arguments": {"query": query}}}
            results = app.toolkit.execute_tool_calls([call])
            return results[0].get("content", "No results") if results else "No results"

        elif tool_name == "current_datetime":
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        else:
            return f"Unknown tool: {tool_name}"

    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"
    except Exception as exc:
        return f"Error: {exc}"


def run_text_tool_loop(
    app: "GemApp",
    user_text: str,
    composed_messages: list[dict],
    out: "OutputManager",
    max_rounds: int = 15,
) -> str:
    """Codex-style agent loop. Model calls tools until it stops on its own.

    The model stops by responding with plain text (no JSON tool call).
    No "done" signal needed — just like Codex's architecture.
    """
    # Inject tool prompt into system message
    messages = list(composed_messages)
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": messages[0]["content"] + "\n\n" + TOOL_PROMPT}
    else:
        messages.insert(0, {"role": "system", "content": TOOL_PROMPT})

    for round_num in range(max_rounds):
        # Show progress
        out.set_stage(f"round {round_num + 1}" if round_num > 0 else "processing")
        out.print_info(f"▶ {'processing' if round_num == 0 else f'round {round_num + 1}'}")

        # Call model — no format constraint, no thinking, just generate
        response = app.engine.chat_once(messages, think=False)
        content = response.get("message", {}).get("content", "").strip()

        # Clean special tokens
        if "<|" in content or "|>" in content:
            content = re.sub(r'<\|[^>]*\|>', '', content).strip()

        out.feed_thinking(content)

        if not content:
            break

        # Try to parse a tool call
        tool_call = parse_tool_call(content)

        if tool_call:
            # ── Tool call: execute and loop ──
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            preview = str(tool_args.get("path", tool_args.get("command", tool_args.get("query", ""))))[:60]
            out.log_tool(tool_name, preview)

            result = execute_tool(app, tool_name, tool_args)
            is_err = result.startswith("Error")
            out.tool_result(result[:120], error=is_err)

            # Append to conversation and continue
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Tool result:\n{result}"})
            out.start_thinking(reset=False)
        else:
            # ── Plain text: model is done ──
            out.stream(content)
            return content

    return ""
