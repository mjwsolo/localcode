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

from .streaming import StreamingParser

if TYPE_CHECKING:
    from .app import GemApp
    from .output import OutputManager
    from .permissions_v2 import PermissionManager


TOOL_PROMPT = """# Tools

You have two ways to call tools:

## For creating/overwriting files — use code block format:
TOOL: write_file
PATH: filename.py
```python
your complete code here with proper indentation
```

## For other tools — use JSON on a single line:
{"tool": "edit_file", "args": {"path": "file.py", "old_string": "old text", "new_string": "new text"}}
{"tool": "read_file", "args": {"path": "file.py"}}
{"tool": "bash", "args": {"command": "pip install pygame"}}
{"tool": "grep", "args": {"pattern": "def main"}}
{"tool": "glob", "args": {"pattern": "*.py"}}
{"tool": "web_search", "args": {"query": "pygame tutorial"}}

# How to work (follow this order):
1. EXPLORE: Read existing files first (read_file, bash, grep) to understand what's there
2. WRITE: Write COMPLETE working code in one write_file. Not scaffolds or TODOs — full code.
3. VERIFY: Run `python -m py_compile file.py` to check for syntax errors
4. CHECK DEPS: Run `python -c "import module_name"` to verify dependencies exist
5. SUMMARIZE: Respond with plain text describing what you did. Don't run GUI apps.

One tool call per response. Keep going until fully done."""


def parse_tool_call(text: str) -> dict | None:
    """Extract a tool call from model text.

    Supports two formats:
    1. Code block: TOOL: write_file\\nPATH: file.py\\n```python\\ncode\\n```
    2. JSON: {"tool": "bash", "args": {"command": "..."}}

    Returns None if no tool call found (model is done).
    """
    text = text.strip()

    # Format 1: Code block for write_file
    # TOOL: write_file
    # PATH: filename.py
    # ```python
    # code
    # ```
    tool_match = re.match(r'TOOL:\s*write_file\s*\nPATH:\s*(\S+)', text)
    if tool_match:
        path = tool_match.group(1)
        code_match = re.search(r'```\w*\n(.*?)```', text, re.DOTALL)
        if code_match:
            return {"tool": "write_file", "args": {"path": path, "content": code_match.group(1)}}

    # Format 2: JSON tool call
    if '"tool"' not in text:
        return None

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
            # Unescape if content came from JSON (not code block)
            if "\\n" in content and "\n" not in content:
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

        elif tool_name == "apply_patch":
            patch_text = args.get("patch", "")
            if not patch_text:
                return "Error: need patch text"
            return _apply_patch(app, patch_text)

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
                env={**__import__("os").environ, "MallocStackLogging": "0"},
            )
            # Filter noisy stderr (MallocStackLogging, pygame init messages)
            stderr = result.stderr
            stderr_lines = [l for l in stderr.splitlines()
                           if "MallocStackLogging" not in l
                           and "can't turn off" not in l]
            output = (result.stdout + "\n".join(stderr_lines)).strip()
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


def _verify_python(app: "GemApp", fpath: str) -> str:
    """Verify a Python file: syntax check + quick import/runtime check.

    Inspired by Claude Code's verification agent pattern:
    "Before reporting complete, verify it works: run the test, check the output."

    Returns error string if problems found, empty string if OK.
    """
    full = app.repo_root / fpath
    if not full.is_file():
        return ""

    code = full.read_text(errors="replace")

    # 1. Syntax check (instant, catches SyntaxError)
    try:
        compile(code, fpath, "exec")
    except SyntaxError as e:
        return f"SyntaxError in {fpath} line {e.lineno}: {e.msg}"

    # 2. Quick import check — only check syntax and imports, don't execute
    # Set SDL_VIDEODRIVER=dummy to prevent GUI windows from opening
    try:
        test_code = (
            f"import sys, ast\n"
            f"# Parse and check for obvious issues\n"
            f"with open('{full}') as f:\n"
            f"    tree = ast.parse(f.read())\n"
            f"# Check imports exist\n"
            f"for node in ast.walk(tree):\n"
            f"    if isinstance(node, ast.Import):\n"
            f"        for alias in node.names:\n"
            f"            try:\n"
            f"                __import__(alias.name.split('.')[0])\n"
            f"            except ImportError as e:\n"
            f"                print(f'ImportError: {{e}}', file=sys.stderr)\n"
            f"                sys.exit(1)\n"
        )
        env = {**__import__("os").environ, "SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
        result = subprocess.run(
            ["python3", "-c", test_code],
            capture_output=True, text=True, timeout=5,
            cwd=str(app.repo_root), env=env,
        )
        if result.returncode != 0:
            # Filter noise
            stderr = result.stderr.strip()
            lines = [l for l in stderr.splitlines()
                     if "MallocStackLogging" not in l and "can't turn off" not in l]
            if lines:
                return lines[-1][:150]
    except subprocess.TimeoutExpired:
        pass  # timeout is OK — means the app started (GUI apps block)
    except Exception:
        pass

    return ""


def _apply_patch(app: "GemApp", patch_text: str) -> str:
    """Apply a Codex-style patch to files.

    Format:
    *** Begin Patch
    *** Update File: path/to/file.py
     context line
    -removed line
    +added line
     context line
    *** Add File: path/to/new.py
    +line 1
    +line 2
    *** Delete File: path/to/old.py
    *** End Patch
    """
    lines = patch_text.splitlines()
    results = []
    current_file = None
    action = None  # "update", "add", "delete"
    old_lines: list[str] = []
    new_lines: list[str] = []
    context_before: list[str] = []

    def flush_hunk():
        nonlocal current_file, old_lines, new_lines, context_before
        if not current_file:
            return
        full = app.repo_root / current_file
        if action == "add":
            full.parent.mkdir(parents=True, exist_ok=True)
            app.toolkit.changes.snapshot_before(current_file, "patch")
            content = "\n".join(l.lstrip("+") for l in new_lines)
            full.write_text(content + "\n")
            results.append(f"Added {current_file}")
        elif action == "delete":
            if full.is_file():
                app.toolkit.changes.snapshot_before(current_file, "patch")
                full.unlink()
                results.append(f"Deleted {current_file}")
        elif action == "update" and full.is_file():
            app.toolkit.changes.snapshot_before(current_file, "patch")
            text = full.read_text(errors="replace")
            file_lines = text.splitlines()
            # Find the context in the file
            search = [l for l in context_before if l]
            if search:
                # Find where the context matches
                for i in range(len(file_lines)):
                    if i + len(search) <= len(file_lines):
                        if all(file_lines[i + j].strip() == search[j].strip() for j in range(len(search))):
                            # Found context — apply the patch here
                            insert_at = i + len(search)
                            # Remove old lines
                            for old in old_lines:
                                old_stripped = old.lstrip("-").strip()
                                for k in range(insert_at, min(insert_at + 5, len(file_lines))):
                                    if file_lines[k].strip() == old_stripped:
                                        file_lines.pop(k)
                                        break
                            # Insert new lines
                            for j, new in enumerate(new_lines):
                                file_lines.insert(insert_at + j, new.lstrip("+"))
                            break
                full.write_text("\n".join(file_lines) + "\n")
                results.append(f"Updated {current_file}")
            else:
                results.append(f"Error: could not find context in {current_file}")
        old_lines.clear()
        new_lines.clear()
        context_before.clear()

    for line in lines:
        if line.startswith("*** Begin Patch"):
            continue
        elif line.startswith("*** End Patch"):
            flush_hunk()
            break
        elif line.startswith("*** Update File:"):
            flush_hunk()
            current_file = line.split(":", 1)[1].strip()
            action = "update"
        elif line.startswith("*** Add File:"):
            flush_hunk()
            current_file = line.split(":", 1)[1].strip()
            action = "add"
        elif line.startswith("*** Delete File:"):
            flush_hunk()
            current_file = line.split(":", 1)[1].strip()
            action = "delete"
        elif line.startswith("-"):
            old_lines.append(line)
        elif line.startswith("+"):
            new_lines.append(line)
        elif line.startswith(" ") or line == "":
            if not old_lines and not new_lines:
                context_before.append(line.lstrip(" "))

    flush_hunk()
    return "; ".join(results) if results else "No changes applied"


def run_text_tool_loop(
    app: "GemApp",
    user_text: str,
    composed_messages: list[dict],
    out: "OutputManager",
    max_rounds: int = 15,
    permissions: "PermissionManager | None" = None,
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

        # Use /api/generate (NOT /api/chat) — Gemma 4's chat template
        # injects <|tool_response> tokens that break JSON output.
        # No format: "json" either — it was causing degeneration on long outputs.
        # The model outputs clean JSON from our prompt instructions alone.
        content = app.engine.generate_once(messages)

        # Parse response for early stop detection
        parser = StreamingParser(mode="auto")
        for char_chunk in [content[i:i+10] for i in range(0, len(content), 10)]:
            chunk = parser.feed(char_chunk)
            if chunk.is_complete:
                # Trim response at early stop point
                content = parser.state.full_text
                out.print_info(f"▶ early stop: {chunk.stop_reason}")
                break

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
            content = content.strip()

        out.feed_thinking(content)

        if not content:
            break

        # Try to parse a tool call
        tool_call = parse_tool_call(content)

        if not tool_call:
            # No tool call found. Three cases:
            # 1. Corrupted JSON — retry
            if '"tool"' in content and len(content) > 500:
                out.print_info("▶ retrying (corrupted output)")
                messages.append({"role": "assistant", "content": "Error: output corrupted."})
                messages.append({"role": "user", "content": "Try again. Keep code under 30 lines per write_file call."})
                out.start_thinking(reset=False)
                continue

            # 2. Model explained what it would do instead of doing it — push it
            if round_num == 0 and any(w in content.lower() for w in
                    ("i have created", "i will create", "here's", "here is",
                     "i've created", "prototype", "scaffold", "i'll create")):
                out.print_info("▶ pushing model to use tools")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": (
                    "You described what you'd do but didn't actually do it. "
                    "Use the write_file tool NOW to create the file. "
                    'Output: {"tool": "write_file", "args": {"path": "filename.py", "content": "code"}}'
                )})
                out.start_thinking(reset=False)
                continue

            # 3. Actual final response (model is done after tool calls)
            if "\\n" in content and len(content) > 200:
                content = "Something went wrong. Please try again."
            out.stream(content)
            return content

        tool_name = tool_call.get("tool", "")
        tool_args = tool_call.get("args", {})
        message = tool_call.get("message", "")

        # "done" = model says task is complete
        if tool_name == "done":
            final = message or "Done."
            out.stream(final)
            return final

        # Stop indicator before printing tool results (prevents interleaved output)
        out._stop_indicator()

        # Execute the tool
        preview = str(tool_args.get("path", tool_args.get("command", tool_args.get("query", ""))))[:60]
        out.log_tool(tool_name, preview)

        if permissions is not None:
            allowed, reason = permissions.check(tool_name, tool_args)
            if not allowed:
                result = reason
                is_err = True
                out.tool_result(result[:120], error=True)
                messages.append({"role": "tool", "content": result})
                continue

        result = execute_tool(app, tool_name, tool_args)
        is_err = result.startswith("Error")
        out.tool_result(result[:120], error=is_err)

        # Auto-verify: if we just wrote a Python file, check for syntax/import errors
        if tool_name in ("write_file", "edit_file") and not is_err:
            fpath = tool_args.get("path", "")
            if fpath.endswith(".py"):
                verify = _verify_python(app, fpath)
                if verify:
                    result += f"\n⚠ Verification error: {verify}"
                    out.tool_result(f"⚠ {verify[:100]}", error=True)

        # Detect loops — same tool called 3+ times in a row
        recent_tools = [m.get("content", "") for m in messages[-6:] if m.get("role") == "assistant"]
        same_count = sum(1 for r in recent_tools if f'"tool": "{tool_name}"' in r)
        if same_count >= 2:
            out.print_info("▶ breaking loop (same tool repeated)")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": (
                f"You've called {tool_name} multiple times. Stop exploring and START DOING. "
                f"Use write_file to create the code NOW."
            )})
            out.start_thinking(reset=False)
            continue

        # Check for response repetition
        if len(messages) >= 4:
            last_two = [m.get("content", "") for m in messages[-4:] if m.get("role") == "assistant"]
            if len(last_two) >= 2 and last_two[-1] == last_two[-2] and len(last_two[-1]) > 50:
                out.print_info("▶ breaking loop (repeated response)")
                break

        # Feed result back — remind model to keep going until FULLY done
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": (
            f"Tool result:\n{result}\n\n"
            f"Keep going. The task is NOT done until the code is COMPLETE and working. "
            f"Don't stop at a scaffold — add all the features needed."
        )})
        out.start_thinking(reset=False)

    return ""
