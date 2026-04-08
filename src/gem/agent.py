"""Agent loop — model-driven tool execution.

The model decides what to do via native Gemma 4 tool calls.
We execute tools and feed results back until the model is done.

Based on: Codex queryLoop (2381 lines), OpenCode agent.go (~300 lines).
Adapted for: Gemma 4 IQ3_S, 16GB Apple Silicon, 32K TurboQuant context.
Proven: 5 turns, 3 files, 100% success rate in testing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markdown import Markdown

if TYPE_CHECKING:
    from .app import GemApp
    from .output import OutputManager


# ── Constants ────────────────────────────────────────────────────────────

MAX_ROUNDS = 20
MAX_OUTPUT_TOKENS = -1  # no limit — model generates until EOS (server has its own safety net)

# Per-tool result size limits (chars)
RESULT_LIMITS = {
    "grep": 20_000,
    "bash": 30_000,
    "read_file": 50_000,
    "web_search": 10_000,
    "default": 50_000,
}
MAX_AGGREGATE_PER_TURN = 100_000

# Destructive commands needing user confirmation
DESTRUCTIVE_PATTERNS = [
    "rm -rf", "rm -r", "rmdir", "git push", "git reset --hard",
    "sudo ", "pip install", "npm install", "brew install",
    "docker rm", "kubectl delete", "DROP TABLE", "DELETE FROM",
]

SYSTEM_PROMPT = """\
You are LocalCode, a coding assistant that runs on the user's machine. You have FULL access to their filesystem via tools.

CRITICAL: You CAN and MUST create, read, and edit files using your tools. You are NOT a plain chatbot. \
You have tools: write_file, edit_file, read_file, bash, grep, glob, list_files, web_search. USE THEM.

Rules:
- ALWAYS use write_file to create files and edit_file to modify them. NEVER tell the user to copy-paste code.
- NEVER paste code in your response text. Use tools to write it directly to disk.
- Keep text responses short — just summarize what you did.
- Work iteratively: write a small working version first, then add features with edit_file.
- Read files before editing. Understand existing code first.
- Prefer editing existing files over creating new ones.
- NEVER leave TODO comments or "I'll fix this later" in code. Write complete, working code.

Working directory: {cwd}
{project_instructions}"""

# Files checked for project-specific instructions (first found wins)
_PROJECT_FILES = ["LOCALCODE.md", "localcode.md", ".localcode.md", ".localcode"]


def _load_project_instructions(repo_root: Path) -> str:
    """Load project-specific instructions from LOCALCODE.md if it exists."""
    for name in _PROJECT_FILES:
        path = repo_root / name
        if path.exists():
            content = path.read_text(errors="replace").strip()
            if content:
                return f"\nProject instructions (from {name}):\n{content}"
    return ""


# ── Tool Schemas ─────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file. You MUST read before editing. Returns content with line numbers.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to repo root"},
            "offset": {"type": "integer", "description": "Start line (0-based). Optional."},
            "limit": {"type": "integer", "description": "Max lines. Default 2000."},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file. Read first if file exists. Prefer edit_file for modifications.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string", "description": "Complete file content"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace exact text in a file. old_string must be unique. Read the file first.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to find (must be unique)"},
            "new_string": {"type": "string", "description": "Replacement text"},
        }, "required": ["path", "old_string", "new_string"]},
    }},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command. Use for: running code, tests, git, installing packages.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents with regex. Returns matches with file paths and line numbers.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Directory to search. Default: repo root."},
            "include": {"type": "string", "description": "File glob filter, e.g. '*.py'"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List directory contents. Skips hidden files and build artifacts.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path. Default: repo root."},
        }},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for documentation, APIs, error solutions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]},
    }},
]


# ── Tool Execution ───────────────────────────────────────────────────────

def _execute_tool(app: "GemApp", name: str, args: dict, out: "OutputManager") -> str:
    """Execute a single tool and return the result string."""
    repo = app.repo_root

    try:
        if name == "read_file":
            return _tool_read_file(repo, args)
        elif name == "write_file":
            return _tool_write_file(repo, args, out)
        elif name == "edit_file":
            return _tool_edit_file(repo, args, out)
        elif name == "bash":
            return _tool_bash(repo, args, out)
        elif name == "grep":
            return _tool_grep(repo, args)
        elif name == "glob":
            return _tool_glob(repo, args)
        elif name == "list_files":
            return _tool_list_files(repo, args)
        elif name == "web_search":
            return _tool_web_search(app, args, out)
        else:
            return f"Unknown tool: {name}. Available: read_file, write_file, edit_file, bash, grep, glob, list_files, web_search"
    except Exception as e:
        return f"Error in {name}: {type(e).__name__}: {e}"


def _tool_read_file(repo: Path, args: dict) -> str:
    if "path" not in args:
        return "Error: 'path' argument is required for read_file."
    path = repo / args["path"]
    if not path.exists():
        return f"File not found: {args['path']}"
    content = path.read_text(errors="replace")
    lines = content.splitlines()
    offset = args.get("offset", 0)
    limit = args.get("limit", 2000)
    selected = lines[offset:offset + limit]
    numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
    result = "\n".join(numbered)
    if len(lines) > offset + limit:
        result += f"\n\n[{len(lines) - offset - limit} more lines — use offset={offset + limit} to continue]"
    return result


def _tool_write_file(repo: Path, args: dict, out: "OutputManager") -> str:
    if "path" not in args:
        return "Error: 'path' argument is required for write_file."
    path = repo / args["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    content = args.get("content", "")
    path.write_text(content)
    lines = content.count("\n") + 1
    return f"Written {args['path']} ({lines} lines)"


def _tool_edit_file(repo: Path, args: dict, out: "OutputManager") -> str:
    if "path" not in args:
        return "Error: 'path' argument is required for edit_file."
    path = repo / args["path"]
    if not path.exists():
        return f"File not found: {args['path']}"
    content = path.read_text(errors="replace")
    old = args["old_string"]
    new = args["new_string"]
    if old not in content:
        # Help the model by showing nearby content
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if old[:30] in line:
                context = "\n".join(lines[max(0, i-2):i+3])
                return f"old_string not found. Similar content near line {i+1}:\n{context}"
        return f"old_string not found in {args['path']}"
    content = content.replace(old, new, 1)
    path.write_text(content)
    return f"Edited {args['path']}: replaced {len(old)} → {len(new)} chars"


def _tool_bash(repo: Path, args: dict, out: "OutputManager") -> str:
    cmd = args["command"]
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=30, cwd=str(repo),
            env={**__import__("os").environ, "MallocStackLogging": "0"},
        )
        output = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            output = f"[exit code {r.returncode}]\n{output}"
        return output or "all good!"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s). If running a GUI app, use 'nohup python app.py &' instead."


def _tool_grep(repo: Path, args: dict) -> str:
    pattern = args["pattern"]
    search_path = str(repo / args.get("path", "."))
    include = args.get("include", "")
    cmd = ["grep", "-rn", "--exclude-dir=.git", "--exclude-dir=.venv",
           "--exclude-dir=node_modules", "--exclude-dir=__pycache__"]
    if include:
        cmd.append(f"--include={include}")
    cmd.extend([pattern, search_path])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        result = r.stdout.strip()
        # Make paths relative
        result = result.replace(str(repo) + "/", "")
        lines = result.splitlines()
        if len(lines) > 50:
            result = "\n".join(lines[:50]) + f"\n\n[{len(lines) - 50} more matches]"
        return result or "No matches found"
    except subprocess.TimeoutExpired:
        return "Error: search timed out"


def _tool_glob(repo: Path, args: dict) -> str:
    pattern = args["pattern"]
    matches = sorted(repo.glob(pattern), key=lambda f: f.stat().st_mtime if f.exists() else 0, reverse=True)
    results = [str(m.relative_to(repo)) for m in matches[:100]
               if not any(p in str(m) for p in (".git", "__pycache__", "node_modules", ".venv"))]
    return "\n".join(results) or "No matches"


def _tool_list_files(repo: Path, args: dict) -> str:
    target = repo / args.get("path", ".")
    if not target.exists():
        return f"Directory not found: {args.get('path', '.')}"
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", "dist", "build"}
    entries = []
    for p in sorted(target.iterdir()):
        if p.name.startswith(".") and p.name != ".env":
            continue
        if p.name in skip:
            continue
        marker = "/" if p.is_dir() else ""
        entries.append(f"  {p.name}{marker}")
    return "\n".join(entries[:200]) or "Empty directory"


def _tool_web_search(app: "GemApp", args: dict, out: "OutputManager") -> str:
    query = args["query"]
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found"
        formatted = []
        for r in results:
            formatted.append(f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}\n")
        return "\n".join(formatted)
    except Exception as e:
        return f"Search error: {e}"


# ── Result Management ────────────────────────────────────────────────────

def _truncate_result(result: str, tool_name: str) -> str:
    """Truncate tool result to per-tool limit."""
    limit = RESULT_LIMITS.get(tool_name, RESULT_LIMITS["default"])
    if len(result) <= limit:
        return result
    half = limit // 2
    return result[:half] + f"\n\n[...{len(result) - limit} chars truncated...]\n\n" + result[-half:]


def _needs_confirmation(name: str, args: dict) -> bool:
    """Check if this tool needs user confirmation."""
    if name != "bash":
        return False
    cmd = args.get("command", "")
    return any(p in cmd for p in DESTRUCTIVE_PATTERNS)


# ── Context Management ───────────────────────────────────────────────────

def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: chars / 4."""
    total = 0
    for m in messages:
        total += len(str(m.get("content", "")))
        for tc in m.get("tool_calls", []):
            total += len(str(tc.get("function", {}).get("arguments", "")))
    return total // 4


def _compact_messages(messages: list[dict], out: "OutputManager") -> list[dict]:
    """Summarize old messages, keep recent context."""
    if len(messages) <= 10:
        return messages

    system = [m for m in messages[:2] if m.get("role") == "system"]
    recent = messages[-6:]

    old = messages[len(system):-6]
    if not old:
        return messages

    # Build summary from old messages
    parts = []
    files_modified = set()
    commands_run = []

    for m in old:
        role = m.get("role", "")
        content = str(m.get("content", ""))

        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except:
                    args = {}
                if name in ("write_file", "edit_file"):
                    files_modified.add(args.get("path", "?"))
                elif name == "bash":
                    commands_run.append(args.get("command", "")[:60])
                parts.append(f"Called {name}({_summarize_args(args)})")

        elif role == "user" and not content.startswith("Previous"):
            parts.append(f"User: {content[:100]}")

    summary_lines = ["Previous conversation summary:"]
    if files_modified:
        summary_lines.append(f"Files modified: {', '.join(files_modified)}")
    if commands_run:
        summary_lines.append(f"Commands run: {'; '.join(commands_run[:5])}")
    summary_lines.extend(parts[-8:])

    out.print_info("Context compacted — older messages summarized.")

    return [
        *system,
        {"role": "user", "content": "\n".join(summary_lines)},
        {"role": "assistant", "content": "Got it. Continuing from where we left off."},
        *recent,
    ]


# ── Display Helpers ──────────────────────────────────────────────────────

def _summarize_args(args: dict) -> str:
    """Short summary of tool args for display."""
    if "path" in args:
        return args["path"]
    if "pattern" in args:
        return args["pattern"]
    if "command" in args:
        return args["command"][:60]
    if "query" in args:
        return args["query"][:40]
    return str(args)[:60]


def _render_markdown(text: str, console: Console | None = None) -> None:
    """Render text as markdown if it has markdown markers, plain otherwise."""
    text = text.strip()
    if not text:
        return
    c = console or Console(width=min(100, __import__("shutil").get_terminal_size().columns))
    has_md = any(m in text for m in ("```", "###", "**", "- ", "1. ", "`"))
    if has_md:
        c.print(Markdown(text))
    else:
        c.print(text)


def _brief_result(tool_name: str, result: str) -> str:
    """Short summary of a tool result for terminal display."""
    lines = result.strip().splitlines()
    if tool_name == "read_file":
        return f"{len(lines)} lines"
    if tool_name == "write_file":
        return result.strip()[:80] if result.strip() else "written"
    if tool_name == "edit_file":
        return result.strip()[:80] if result.strip() else "edited"
    if tool_name == "bash":
        if not result.strip():
            return "done (no output)"
        if len(lines) <= 3:
            return "\n".join(lines)[:200]
        return f"{lines[0][:80]}  …({len(lines)} lines)"
    if tool_name == "grep":
        return f"{len(lines)} matches" if lines and lines[0] else "no matches"
    if tool_name in ("glob", "list_files"):
        return f"{len(lines)} files"
    if tool_name == "web_search":
        return f"{len(lines) // 3} results" if lines else "no results"
    return result[:80] if result else "done"


def _tool_stage_label(tool_name: str, args: dict) -> str:
    """Human-readable stage label for the indicator."""
    if tool_name == "bash":
        cmd = args.get("command", "")
        if "pip " in cmd:
            return "installing packages"
        if "npm " in cmd:
            return "installing packages"
        if "git " in cmd:
            return "git operation"
        if "python " in cmd or "pytest" in cmd:
            return "running code"
        return f"running command"
    if tool_name == "write_file":
        return f"writing {Path(args.get('path', 'file')).name}"
    if tool_name == "edit_file":
        return f"editing {Path(args.get('path', 'file')).name}"
    if tool_name == "read_file":
        return f"reading {Path(args.get('path', 'file')).name}"
    if tool_name == "grep":
        return "searching code"
    if tool_name == "glob":
        return "finding files"
    if tool_name == "web_search":
        return "searching web"
    return tool_name


# ── Main Agent Loop ──────────────────────────────────────────────────────

def run_agent_loop(
    app: "GemApp",
    user_text: str,
    composed_messages: list[dict],
    out: "OutputManager",
) -> str:
    """Model-driven agent loop.

    The model decides what to do via native Gemma 4 tool calls.
    We execute tools and feed results back until the model is done.

    Returns the final text response from the model.
    """
    from .runtime import _strip_thinking_tokens

    # ── Build messages ──
    # composed_messages already has system prompt + context + full conversation + current user msg
    # Inject our tool-loop system prompt at the front
    project_instructions = _load_project_instructions(app.repo_root)
    agent_system = SYSTEM_PROMPT.format(cwd=app.repo_root, project_instructions=project_instructions)
    messages: list[dict[str, Any]] = []

    # Our agent prompt goes first — it's the most important
    messages.append({"role": "system", "content": agent_system})

    for m in composed_messages:
        if m.get("role") == "system":
            continue  # skip old system prompt — ours is authoritative
        messages.append(m)

    # Add current user message if not already there
    if not messages or messages[-1].get("content") != user_text:
        messages.append({"role": "user", "content": user_text})

    use_thinking = app.config.runtime.laptop_26b_runtime_mode.endswith("-think")
    full_response: list[str] = []
    start_time = time.time()
    tools_called: list[str] = []
    recent_tool_sigs: list[str] = []  # for loop detection
    loop_detected = False

    # ── Main loop ──
    for round_num in range(MAX_ROUNDS):
        # Stream model response — text appears live, tool calls collected
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict] = []
        thinking_shown = False

        try:
            for event in app.engine.stream_chat_events(
                messages, tools=TOOL_SCHEMAS, think=use_thinking, num_predict=MAX_OUTPUT_TOKENS,
            ):
                if event["type"] == "thinking":
                    chunk = _strip_thinking_tokens(event["content"])
                    if chunk:
                        if not thinking_shown:
                            out._stop_indicator()
                            sys.stdout.write("\033[2;3m  thinking...\033[0m\n")
                            thinking_shown = True
                        thinking_parts.append(chunk)
                elif event["type"] == "content":
                    chunk = _strip_thinking_tokens(event["content"])
                    if chunk:
                        content_parts.append(chunk)
                elif event["type"] == "tool_calls":
                    tool_calls = event["tool_calls"]
        except KeyboardInterrupt:
            out.print_info("Interrupted.")
            break
        except Exception as exc:
            # Retry once without thinking
            try:
                for event in app.engine.stream_chat_events(
                    messages, tools=TOOL_SCHEMAS, think=False, num_predict=MAX_OUTPUT_TOKENS,
                ):
                    if event["type"] == "content":
                        chunk = _strip_thinking_tokens(event["content"])
                        if chunk:
                            content_parts.append(chunk)
                    elif event["type"] == "tool_calls":
                        tool_calls = event["tool_calls"]
            except Exception:
                out.set_error(f"Model error: {exc}")
                break

        # Show thinking summary if present (collapsed, dim)
        if thinking_parts:
            thinking_text = "".join(thinking_parts).strip()
            if thinking_text:
                # Show first line as a peek, truncated to terminal width
                lines = thinking_text.splitlines()
                cols = __import__("shutil").get_terminal_size().columns
                max_len = cols - 16  # account for "  thought: " + margin
                preview = lines[0][:max_len]
                if len(lines) > 1 or len(lines[0]) > max_len:
                    preview = preview[:max_len - 3] + "…"
                sys.stdout.write(f"\033[2;3m  thought: {preview}\033[0m\n")

        content = "".join(content_parts)

        # Clear the indicator before rendering output
        out._stop_indicator()
        sys.stdout.write("\r\033[K")  # clear indicator line
        sys.stdout.flush()

        # Build assistant message for history
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # ── No tool calls = model is done ──
        if not tool_calls:
            if content:
                _render_markdown(content, app.console if hasattr(app, 'console') else None)
                full_response.append(content)
            break

        # ── Execute tools ──
        aggregate_size = 0

        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "unknown")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
                out.print_info(f"Warning: malformed args for {tool_name}")

            # Update indicator immediately
            stage = _tool_stage_label(tool_name, args)
            out.set_stage(stage)
            idx = out.log_tool(tool_name, _summarize_args(args))

            # Safety: confirm destructive commands
            if _needs_confirmation(tool_name, args):
                import tty, termios
                cmd = args.get("command", "")
                sys.stdout.write(f"\n\033[33m    ┌ Allow this command?\033[0m\n")
                sys.stdout.write(f"\033[2m    │ {cmd[:70]}\033[0m\n")
                sys.stdout.write(f"    │\n")
                sys.stdout.write(f"    │  \033[1m1\033[0m  yes, run it\n")
                sys.stdout.write(f"    │  \033[1m2\033[0m  no, skip\n")
                sys.stdout.write(f"\033[33m    └\033[0m ")
                sys.stdout.flush()
                try:
                    fd = sys.stdin.fileno()
                    old = termios.tcgetattr(fd)
                    try:
                        tty.setraw(fd)
                        ch = sys.stdin.read(1)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
                except Exception:
                    try:
                        ch = input().strip()
                    except EOFError:
                        ch = "2"
                if ch == "2" or ch == "n":
                    sys.stdout.write(f"\033[2mskipped\033[0m\n")
                    messages.append({"role": "tool", "content": "Denied by user.", "tool_call_id": tc.get("id", "")})
                    continue
                sys.stdout.write(f"\033[32mrunning\033[0m\n")

            # Loop detection — break if same tool+args repeats 3x
            sig = f"{tool_name}:{_summarize_args(args)}"
            recent_tool_sigs.append(sig)
            if len(recent_tool_sigs) >= 3 and recent_tool_sigs[-1] == recent_tool_sigs[-2] == recent_tool_sigs[-3]:
                out.print_info("Loop detected — stopping repeated tool calls.")
                messages.append({"role": "tool", "content": "STOP. You already called this tool 3 times with the same arguments. Summarize what you did and finish.", "tool_call_id": tc.get("id", "")})
                loop_detected = True
                break

            # Execute
            tool_result = _execute_tool(app, tool_name, args, out)
            tools_called.append(tool_name)

            # Show result to user
            is_error = tool_result.startswith("Error:") or tool_result.startswith("error:")
            out.tool_result(_brief_result(tool_name, tool_result), error=is_error, idx=idx)

            # Truncate per-tool
            tool_result = _truncate_result(tool_result, tool_name)
            aggregate_size += len(tool_result)

            # Aggregate budget
            if aggregate_size > MAX_AGGREGATE_PER_TURN:
                tool_result = tool_result[:500] + "\n[Truncated — context budget exceeded this turn]"

            # Add to history
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tc.get("id", ""),
            })

        # Render any text between tool rounds
        if content:
            _render_markdown(content)
            full_response.append(content)

        # Break outer loop if loop was detected
        if loop_detected:
            break

        # Context compaction check (85% of 32K = ~27K tokens)
        if _estimate_tokens(messages) > 27_000:
            messages = _compact_messages(messages, out)

    else:
        out.print_info(f"Reached max rounds ({MAX_ROUNDS})")

    # ── Summary ──
    elapsed = time.time() - start_time
    if tools_called and elapsed > 3:
        tool_summary = ", ".join(f"{t}×{tools_called.count(t)}" for t in dict.fromkeys(tools_called))
        out.print_info(f"Done in {elapsed:.0f}s — tools: {tool_summary}")

    return "".join(full_response).strip()
