from __future__ import annotations

import difflib
import fnmatch
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from .config import AppConfig
from .context import IGNORE_DIRS, list_repo_files
from .indexer import build_index, search_index
# MCP tools are lazily wired in via ensure_mcp_tools() (the `mcp` module is
# imported there, on demand, so a missing/broken MCP setup never affects
# toolkit import). See ensure_mcp_tools() below.
from .shell import run_shell
from .undo import ChangeLog

ToolHandler = Callable[[dict[str, Any]], str]


@dataclass
class LocalCodeTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class LocalCodeToolkit:
    # Exit codes that are NOT errors for specific commands
    COMMAND_EXIT_SEMANTICS: dict[str, dict[int, str]] = {
        "grep": {1: "no matches (not an error)"},
        "rg": {1: "no matches (not an error)"},
        "diff": {1: "files differ (not an error)"},
        "test": {1: "condition false (not an error)"},
        "find": {1: "no matches (not an error)"},
    }

    def __init__(self, repo_root: Path, config: AppConfig, app: Any = None) -> None:
        self.repo_root = repo_root
        self.config = config
        self.app = app
        self.tools: dict[str, LocalCodeTool] = {}
        self.plugin_errors: list[str] = []
        self.changes = ChangeLog(repo_root)
        # MCP wiring was deferred out during the T0.9 purge, but close()
        # and diagnostics() still reference these — initialize them so
        # those methods don't AttributeError. close() runs on every
        # session teardown (app.py), so the crash was guaranteed.
        self.mcp_clients: dict[str, Any] = {}
        self._mcp_loaded: bool = False
        self.mcp_errors: list[str] = []
        # File state tracking (agent pattern: pre-read validation + staleness)
        self._read_state: dict[str, float] = {}  # path → timestamp when last read
        self._register_builtin_tools()
        self._register_plugin_tools()

    def _register(self, tool: LocalCodeTool) -> None:
        self.tools[tool.name] = tool

    # ── Built-in tools ───────────────────────────────────────────────────

    def _register_builtin_tools(self) -> None:
        # -- File operations --
        self._register(LocalCodeTool(
            name="read_file",
            description=(
                "Read a file from the repository. Returns file content with line numbers. "
                "Use offset and limit to read specific portions of large files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "offset": {"type": "integer", "description": "Line number to start from (0-based). Optional."},
                    "limit": {"type": "integer", "description": "Max number of lines to return. Optional, default 500."},
                },
                "required": ["path"],
            },
            handler=lambda args: self._read_file(
                str(args["path"]),
                offset=int(args.get("offset", 0)),
                limit=int(args.get("limit", 500)),
            ),
        ))

        self._register(LocalCodeTool(
            name="write_file",
            description="Create a new file or completely overwrite an existing file with new content.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "content": {"type": "string", "description": "Complete file content to write"},
                },
                "required": ["path", "content"],
            },
            handler=lambda args: self._write_file(str(args["path"]), str(args["content"])),
        ))

        self._register(LocalCodeTool(
            name="edit_file",
            description=(
                "Make a surgical edit to a file by replacing an exact string match. "
                "Provide old_string (the exact text to find) and new_string (replacement). "
                "old_string must uniquely match one location in the file. "
                "For creating new files, use write_file instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "old_string": {"type": "string", "description": "Exact text to find and replace (must be unique in file)"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            handler=lambda args: (
                self._edit_file(str(args["path"]), str(args["old_string"]), str(args["new_string"]))
                if "old_string" in args and "new_string" in args
                else f"Error: edit_file requires 'path', 'old_string', and 'new_string'. You provided: {list(args.keys())}. Use read_file first to see the exact text, then provide old_string (exact text to find) and new_string (replacement)."
            ),
        ))

        self._register(LocalCodeTool(
            name="multi_edit",
            description=(
                "Apply multiple edits to a single file atomically. Each edit is an {old_string, new_string} pair. "
                "All old_strings must be unique. Edits are applied in order."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                            },
                            "required": ["old_string", "new_string"],
                        },
                        "description": "List of {old_string, new_string} pairs to apply",
                    },
                },
                "required": ["path", "edits"],
            },
            handler=lambda args: self._multi_edit(str(args["path"]), args["edits"]),
        ))

        # -- Search and navigation --
        self._register(LocalCodeTool(
            name="grep",
            description=(
                "Search file contents using regex. Returns matching lines with file paths and line numbers. "
                "Fast local search across the entire repository or a specific path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search in (relative). Default: repo root."},
                    "include": {"type": "string", "description": "Glob pattern to filter files, e.g. '*.py', '*.ts'. Optional."},
                    "max_results": {"type": "integer", "description": "Max matches to return. Default 50."},
                    "case_insensitive": {"type": "boolean", "description": "Case-insensitive search. Default false."},
                },
                "required": ["pattern"],
            },
            handler=lambda args: self._grep(
                str(args["pattern"]),
                path=str(args.get("path", "")),
                include=str(args.get("include", "")),
                max_results=int(args.get("max_results", 50)),
                case_insensitive=bool(args.get("case_insensitive", False)),
            ),
        ))

        self._register(LocalCodeTool(
            name="glob",
            description=(
                "Find files matching a glob pattern. Returns file paths sorted by modification time. "
                "Supports patterns like '**/*.py', 'src/**/*.ts', '*.json'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or 'src/*.ts'"},
                    "path": {"type": "string", "description": "Directory to search in (relative). Default: repo root."},
                    "max_results": {"type": "integer", "description": "Max files to return. Default 100."},
                },
                "required": ["pattern"],
            },
            handler=lambda args: self._glob(
                str(args["pattern"]),
                path=str(args.get("path", "")),
                max_results=int(args.get("max_results", 100)),
            ),
        ))

        self._register(LocalCodeTool(
            name="list_files",
            description="List repository files, optionally filtered by a substring pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Substring to filter file paths. Optional."},
                },
            },
            handler=lambda args: "\n".join(
                list_repo_files(self.repo_root, args.get("pattern"), limit=300)
            ) or "No files matched.",
        ))

        self._register(LocalCodeTool(
            name="search_code",
            description="Search the local code index for semantically relevant code chunks.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=lambda args: self._search_code(str(args["query"])),
        ))

        # -- Shell execution --
        self._register(LocalCodeTool(
            name="bash",
            description=(
                "Execute a shell command and return its output. "
                "Use for running tests, builds, git commands, installing packages, etc. "
                "Commands run in the repository root with a 120s timeout. "
                "Prefer dedicated tools (read_file, edit_file, grep) over shell equivalents."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds. Default 120."},
                },
                "required": ["command"],
            },
            handler=lambda args: self._bash(
                str(args["command"]),
                timeout=int(args.get("timeout", 120)),
            ),
        ))

        # -- Tool search (deferred tool loading, like agent) --
        self._register(LocalCodeTool(
            name="tool_search",
            description=(
                "Search for additional tools not in your current set. "
                "Use when you need a capability not listed in your tools. "
                f"Available deferred tools: {', '.join(sorted(set(self.tools.keys()) - self.CORE_TOOLS))}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Tool name or keyword to search for"},
                },
                "required": ["query"],
            },
            handler=lambda args: self._tool_search(str(args["query"])),
        ))

        # -- Security review --
        self._register(LocalCodeTool(
            name="security_scan",
            description=(
                "Scan code for security vulnerabilities: hardcoded secrets, SQL injection, "
                "command injection, XSS, insecure crypto, exposed credentials."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or directory to scan. Default: whole repo."},
                },
            },
            handler=lambda args: self._security_scan(str(args.get("path", ""))),
        ))

        # Sub-agent delegation tool removed — the handler was a stub that
        # only returned "delegation has been removed", so exposing it just
        # spent context tokens on a tool the model could never use.

        # -- Codemod (regex replace across files) --
        self._register(LocalCodeTool(
            name="codemod",
            description=(
                "Find and replace a regex pattern across all matching files. "
                "Returns a preview of changes. Useful for renames and refactoring."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to find"},
                    "replacement": {"type": "string", "description": "Replacement string ($1 for groups)"},
                    "include": {"type": "string", "description": "Glob to filter files, e.g. '*.py'. Default: all files."},
                },
                "required": ["pattern", "replacement"],
            },
            handler=lambda args: self._codemod(
                str(args["pattern"]),
                str(args["replacement"]),
                include=str(args.get("include", "")),
            ),
        ))

        # -- Test runner --
        self._register(LocalCodeTool(
            name="run_tests",
            description=(
                "Detect and run the project's test suite. Returns pass/fail results. "
                "Auto-detects pytest, npm test, cargo test, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Override test command. Leave empty to auto-detect."},
                },
            },
            handler=lambda args: self._run_tests(command=str(args.get("command", ""))),
        ))

        # -- REPL (execute code snippets) --
        self._register(LocalCodeTool(
            name="repl",
            description=(
                "Execute a code snippet and return the output. "
                "Supports python and shell. Use to test ideas without creating files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute"},
                    "language": {"type": "string", "description": "python or shell. Default python."},
                },
                "required": ["code"],
            },
            handler=lambda args: self._repl(
                str(args["code"]),
                lang=str(args.get("language", "python")),
            ),
        ))

        # -- System info --
        self._register(LocalCodeTool(
            name="current_datetime",
            description="Get the current date and time. Use this for any time/date questions.",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: self._current_datetime(),
        ))

        # -- Git operations --
        self._register(LocalCodeTool(
            name="git_status",
            description="Show the current git status (modified, staged, untracked files).",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: self._git_command(["git", "status", "--short"]),
        ))

        self._register(LocalCodeTool(
            name="git_diff",
            description="Show the current working tree diff, or diff for specific files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Specific file path to diff. Optional."},
                    "staged": {"type": "boolean", "description": "Show staged changes. Default false."},
                },
            },
            handler=lambda args: self._git_diff_tool(
                path=str(args.get("path", "")),
                staged=bool(args.get("staged", False)),
            ),
        ))

        self._register(LocalCodeTool(
            name="git_log",
            description="Show recent git commit history.",
            parameters={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of commits. Default 20."},
                    "path": {"type": "string", "description": "Show history for a specific file. Optional."},
                    "oneline": {"type": "boolean", "description": "Compact one-line format. Default true."},
                },
            },
            handler=lambda args: self._git_log(
                count=int(args.get("count", 20)),
                path=str(args.get("path", "")),
                oneline=bool(args.get("oneline", True)),
            ),
        ))

        self._register(LocalCodeTool(
            name="git_commit",
            description="Stage files and create a git commit.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"},
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to stage. If empty, commits currently staged files.",
                    },
                },
                "required": ["message"],
            },
            handler=lambda args: self._git_commit(
                str(args["message"]),
                files=list(args.get("files", [])),
            ),
        ))

        # -- Web search --
        self._register(LocalCodeTool(
            name="web_search",
            description="Search the web for current information. Returns titles, snippets, and URLs.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results. Default 5."},
                },
                "required": ["query"],
            },
            handler=lambda args: self._web_search(
                str(args["query"]),
                max_results=int(args.get("max_results", 5)),
            ),
        ))

        self._register(LocalCodeTool(
            name="web_fetch",
            description="Fetch the text content of a web page URL. Useful for reading documentation.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Max chars to return. Default 12000."},
                },
                "required": ["url"],
            },
            handler=lambda args: self._web_fetch(
                str(args["url"]),
                max_chars=int(args.get("max_chars", 12000)),
            ),
        ))

    # ── Tool implementations ─────────────────────────────────────────────

    MAX_FILE_SIZE = 1024 * 1024  # 1MB limit (agent uses 1GB, we're conservative for local)

    def _read_file(self, relative_path: str, offset: int = 0, limit: int = 500) -> str:
        path = (self.repo_root / relative_path).resolve()
        if not path.is_file():
            return f"File not found: {relative_path}"
        # Size check
        try:
            size = path.stat().st_size
            if size > self.MAX_FILE_SIZE:
                return f"File too large ({size // 1024}KB). Max is {self.MAX_FILE_SIZE // 1024}KB."
        except Exception:
            pass
        # Track that this file was read (for pre-edit validation)
        import time as _time
        self._read_state[relative_path] = _time.time()
        try:
            lines = path.read_text(errors="replace").splitlines(keepends=True)
        except Exception as exc:
            return f"Error reading {relative_path}: {exc}"
        total = len(lines)
        if offset >= total:
            return f"{relative_path}: {total} lines total, offset {offset} is past end of file."
        end = min(offset + limit, total)
        selected = lines[offset:end]
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i:>6}\t{line.rstrip()}")
        result = "\n".join(numbered)
        header = f"{relative_path} ({total} lines)"
        if offset > 0 or end < total:
            header += f" [showing lines {offset + 1}-{end}]"
        return f"{header}\n{result}"

    def _write_file(self, relative_path: str, content: str) -> str:
        path = (self.repo_root / relative_path).resolve()
        if self.repo_root.resolve() not in path.parents and path != self.repo_root.resolve():
            return "Error: refusing to write outside the repo root"
        self.changes.snapshot_before(relative_path, "write_file")
        before = path.read_text(errors="replace") if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if not before:
            return f"Created {relative_path} ({len(content)} chars)"
        diff = "\n".join(
            difflib.unified_diff(
                before.splitlines(), content.splitlines(),
                fromfile=f"a/{relative_path}", tofile=f"b/{relative_path}", lineterm="",
            )
        )
        return f"Wrote {relative_path}\n\n{diff[:4000] or '(file overwritten)'}"

    @staticmethod
    def _normalize_quotes(s: str) -> str:
        """Normalize curly/smart quotes to straight quotes."""
        return s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')

    def _edit_file(self, relative_path: str, old_string: str, new_string: str) -> str:
        path = (self.repo_root / relative_path).resolve()
        if self.repo_root.resolve() not in path.parents and path != self.repo_root.resolve():
            return "Error: refusing to edit outside the repo root"

        # Pre-read validation: model must read the file before editing
        if relative_path not in self._read_state:
            return f"Error: you must read_file('{relative_path}') before editing it. Read first, then edit."

        # Staleness check: file changed since last read?
        if path.is_file():
            current_mtime = path.stat().st_mtime
            read_time = self._read_state.get(relative_path, 0)
            if current_mtime > read_time + 1:  # 1s tolerance
                self._read_state.pop(relative_path, None)
                return f"Error: {relative_path} has been modified since you read it. Read it again first."

        self.changes.snapshot_before(relative_path, "edit_file")
        if not path.is_file():
            return f"File not found: {relative_path}"
        text = path.read_text(errors="replace")
        count = text.count(old_string)

        # Quote normalization fallback: try with normalized quotes
        if count == 0:
            normalized_old = self._normalize_quotes(old_string)
            normalized_text = self._normalize_quotes(text)
            if normalized_text.count(normalized_old) == 1:
                # Find the actual substring in the original text
                idx = normalized_text.index(normalized_old)
                actual_old = text[idx:idx + len(old_string)]
                old_string = actual_old
                count = 1

        if count == 0:
            lines = text.splitlines()
            first_line = old_string.split("\n")[0].strip() if old_string else ""
            nearby = []
            for i, line in enumerate(lines):
                if first_line and first_line in line:
                    start = max(0, i - 2)
                    end = min(len(lines), i + 5)
                    nearby = [f"{j + 1:>6}\t{lines[j]}" for j in range(start, end)]
                    break
            hint = ""
            if nearby:
                hint = "\n\nNearest match context:\n" + "\n".join(nearby)
            return f"Error: old_string not found in {relative_path}{hint}"
        if count > 1:
            return f"Error: old_string matches {count} locations in {relative_path}. Provide more context to make it unique."
        updated = text.replace(old_string, new_string, 1)
        path.write_text(updated)
        # Show a compact diff of what changed
        old_lines = old_string.splitlines()
        new_lines = new_string.splitlines()
        diff = "\n".join(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{relative_path}", tofile=f"b/{relative_path}", lineterm="",
        ))
        return f"Edited {relative_path}\n{diff[:3000]}"

    def _multi_edit(self, relative_path: str, edits: list[dict[str, str]]) -> str:
        path = (self.repo_root / relative_path).resolve()
        if self.repo_root.resolve() not in path.parents and path != self.repo_root.resolve():
            return "Error: refusing to edit outside the repo root"
        self.changes.snapshot_before(relative_path, "multi_edit")
        if not path.is_file():
            return f"File not found: {relative_path}"
        text = path.read_text(errors="replace")
        original = text
        results = []
        for i, edit in enumerate(edits):
            old = str(edit.get("old_string", ""))
            new = str(edit.get("new_string", ""))
            count = text.count(old)
            if count == 0:
                results.append(f"  edit {i + 1}: old_string not found, skipped")
                continue
            if count > 1:
                results.append(f"  edit {i + 1}: old_string matches {count} locations, skipped")
                continue
            text = text.replace(old, new, 1)
            results.append(f"  edit {i + 1}: applied")
        if text != original:
            path.write_text(text)
        applied = sum(1 for r in results if "applied" in r)
        return f"Multi-edit {relative_path}: {applied}/{len(edits)} applied\n" + "\n".join(results)

    def _grep(
        self,
        pattern: str,
        path: str = "",
        include: str = "",
        max_results: int = 50,
        case_insensitive: bool = False,
    ) -> str:
        search_root = (self.repo_root / path).resolve() if path else self.repo_root.resolve()
        if not search_root.exists():
            return f"Path not found: {path}"

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"Invalid regex: {exc}"

        matches: list[str] = []
        files_searched = 0

        def _search_file(file_path: Path) -> None:
            nonlocal files_searched
            if include and not fnmatch.fnmatch(file_path.name, include):
                return
            try:
                content = file_path.read_text(errors="replace")
            except Exception:
                return
            files_searched += 1
            rel = str(file_path.relative_to(self.repo_root))
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{rel}:{line_num}: {line.rstrip()[:200]}")
                    if len(matches) >= max_results:
                        return

        if search_root.is_file():
            _search_file(search_root)
        else:
            for file_path in search_root.rglob("*"):
                if not file_path.is_file():
                    continue
                if any(part in IGNORE_DIRS for part in file_path.relative_to(self.repo_root).parts):
                    continue
                # skip binary files by extension
                if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".zip", ".tar", ".gz", ".bin", ".exe", ".dll", ".so", ".pyc", ".pyo"}:
                    continue
                _search_file(file_path)
                if len(matches) >= max_results:
                    break

        if not matches:
            return f"No matches for /{pattern}/ in {files_searched} files."
        header = f"{len(matches)} matches in {files_searched} files"
        if len(matches) >= max_results:
            header += f" (limited to {max_results})"
        return f"{header}\n" + "\n".join(matches)

    def _glob(self, pattern: str, path: str = "", max_results: int = 100) -> str:
        search_root = (self.repo_root / path).resolve() if path else self.repo_root.resolve()
        if not search_root.exists():
            return f"Path not found: {path}"

        results: list[tuple[float, str]] = []
        for file_path in search_root.rglob(pattern):
            if not file_path.is_file():
                continue
            if any(part in IGNORE_DIRS for part in file_path.relative_to(self.repo_root).parts):
                continue
            try:
                mtime = file_path.stat().st_mtime
            except Exception:
                mtime = 0
            results.append((mtime, str(file_path.relative_to(self.repo_root))))
            if len(results) >= max_results * 2:  # collect extra for sorting
                break

        results.sort(key=lambda x: x[0], reverse=True)
        paths = [r[1] for r in results[:max_results]]

        if not paths:
            return f"No files matching '{pattern}'"
        header = f"{len(paths)} files"
        if len(results) > max_results:
            header += f" (showing {max_results} most recent)"
        return f"{header}\n" + "\n".join(paths)

    def _bash(self, command: str, timeout: int = 120) -> str:
        # Safety: block obviously destructive commands
        stripped = command.strip().lower()
        blocked = ["rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev", ":(){:|:&};:"]
        for b in blocked:
            if stripped.startswith(b):
                return f"Blocked: dangerous command '{command}'"

        result = run_shell(command, str(self.repo_root), timeout=min(timeout, 300))
        output = result.output
        if len(output) > 50000:
            output = output[:25000] + "\n...[middle truncated]...\n" + output[-25000:]
        # Exit code semantics: some non-zero codes aren't errors
        cmd_name = command.split()[0] if command.split() else ""
        semantics = self.COMMAND_EXIT_SEMANTICS.get(cmd_name, {})
        if result.returncode in semantics:
            status = f"exit code: {result.returncode} ({semantics[result.returncode]})"
        elif result.returncode == 0:
            status = "exit code: 0 (success)"
        elif result.timed_out:
            status = f"exit code: {result.returncode} (timed out)"
        else:
            status = f"exit code: {result.returncode} (error)"
        return f"$ {command}\n{output}\n[{status}]"

    def _git_command(self, cmd: list[str], max_chars: int = 20000) -> str:
        try:
            result = subprocess.run(
                cmd, cwd=self.repo_root,
                capture_output=True, text=True, check=False, timeout=30,
            )
            output = (result.stdout + result.stderr).strip()
            if len(output) > max_chars:
                output = output[:max_chars] + "\n...[truncated]"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Git command timed out"
        except Exception as exc:
            return f"Git error: {exc}"

    def _git_diff_tool(self, path: str = "", staged: bool = False) -> str:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")
        cmd.append("--")
        if path:
            cmd.append(path)
        else:
            cmd.append(".")
        return self._git_command(cmd)

    def _git_log(self, count: int = 20, path: str = "", oneline: bool = True) -> str:
        cmd = ["git", "log", f"-{min(count, 100)}"]
        if oneline:
            cmd.append("--oneline")
        else:
            cmd.extend(["--format=%h %an %ad %s", "--date=short"])
        if path:
            cmd.extend(["--", path])
        return self._git_command(cmd)

    def _git_commit(self, message: str, files: list[str] | None = None) -> str:
        if files:
            for f in files:
                add_result = subprocess.run(
                    ["git", "add", f], cwd=self.repo_root,
                    capture_output=True, text=True, check=False, timeout=15,
                )
                if add_result.returncode != 0:
                    return f"git add failed for {f}: {add_result.stderr.strip()}"

        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.repo_root,
            capture_output=True, text=True, check=False, timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return f"Commit failed: {output}"
        return f"Committed: {output}"

    def _tool_search(self, query: str) -> str:
        """Search for deferred tools and return their schemas."""
        query_lower = query.lower()
        matches = []
        for name, tool in self.tools.items():
            if query_lower in name.lower() or query_lower in tool.description.lower():
                matches.append(f"  {name}: {tool.description[:100]}")
        if not matches:
            return f"No tools matching '{query}'. Available: {', '.join(sorted(self.tools.keys()))}"
        # Add matched tools to the active set for this session
        return f"Found {len(matches)} matching tools:\n" + "\n".join(matches[:10])

    def _security_scan(self, path: str = "") -> str:
        """Scan for common security vulnerabilities using regex patterns."""
        import re as _re
        PATTERNS = [
            ("Hardcoded secret/key", r'(?:api[_-]?key|secret|password|token|credential)\s*[=:]\s*["\'][^"\']{8,}["\']'),
            ("SQL injection risk", r'(?:execute|cursor\.execute|query)\s*\(\s*[f"\'].*\{'),
            ("Command injection", r'(?:os\.system|subprocess\.call|subprocess\.run)\s*\(\s*[f"\']'),
            ("Hardcoded password", r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']'),
            ("Insecure HTTP", r'http://(?!localhost|127\.0\.0\.1)'),
            ("Eval usage", r'\beval\s*\('),
            ("Pickle load", r'pickle\.loads?\('),
            (".env file reference", r'\.env\b'),
            ("Private key", r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----'),
            ("AWS key pattern", r'AKIA[0-9A-Z]{16}'),
        ]
        scan_root = (self.repo_root / path).resolve() if path else self.repo_root.resolve()
        findings: list[str] = []
        files_scanned = 0

        for file_path in scan_root.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in IGNORE_DIRS for part in file_path.relative_to(self.repo_root).parts):
                continue
            if file_path.suffix.lower() in {".png", ".jpg", ".zip", ".bin", ".pyc", ".woff"}:
                continue
            try:
                content = file_path.read_text(errors="replace")
            except Exception:
                continue
            files_scanned += 1
            rel = str(file_path.relative_to(self.repo_root))
            for label, pattern in PATTERNS:
                for match in _re.finditer(pattern, content, _re.IGNORECASE):
                    line_num = content[:match.start()].count("\n") + 1
                    snippet = match.group(0)[:60]
                    findings.append(f"  [{label}] {rel}:{line_num} — {snippet}")
                    if len(findings) >= 50:
                        break
                if len(findings) >= 50:
                    break

        if not findings:
            return f"No security issues found in {files_scanned} files."
        return f"Found {len(findings)} potential issues in {files_scanned} files:\n" + "\n".join(findings)

    def _codemod(self, pattern: str, replacement: str, include: str = "") -> str:
        """Regex find-and-replace across all matching files."""
        import re as _re
        try:
            regex = _re.compile(pattern)
        except _re.error as exc:
            return f"Invalid regex: {exc}"

        modified_files: list[str] = []
        total_replacements = 0
        preview_lines: list[str] = []

        for file_path in self.repo_root.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in IGNORE_DIRS for part in file_path.relative_to(self.repo_root).parts):
                continue
            if file_path.suffix.lower() in {".png", ".jpg", ".gif", ".zip", ".bin", ".pyc"}:
                continue
            if include and not fnmatch.fnmatch(file_path.name, include):
                continue
            try:
                content = file_path.read_text(errors="replace")
            except Exception:
                continue
            new_content, count = regex.subn(replacement, content)
            if count > 0:
                rel = str(file_path.relative_to(self.repo_root))
                self.changes.snapshot_before(rel, "codemod")
                file_path.write_text(new_content)
                modified_files.append(rel)
                total_replacements += count
                preview_lines.append(f"  {rel}: {count} replacement(s)")

        if not modified_files:
            return f"No matches for /{pattern}/ in {'*' if not include else include}"
        header = f"Modified {len(modified_files)} file(s), {total_replacements} total replacements"
        return f"{header}\n" + "\n".join(preview_lines[:20])

    def _run_tests(self, command: str = "") -> str:
        """Run the project's test suite."""
        from .verification import run_verification
        output, code = run_verification(self.repo_root, command=command or None)
        status = "PASSED" if code == 0 else "FAILED"
        return f"Tests {status} (exit {code})\n{output}"

    def _repl(self, code: str, lang: str = "python") -> str:
        """Execute a code snippet in a subprocess."""
        import tempfile
        if lang in ("shell", "bash", "sh"):
            result = run_shell(code, str(self.repo_root), timeout=30)
            return f"$ {code}\n{result.output}\n[exit {result.returncode}]"
        # Python
        tmp = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir=str(self.repo_root))
        try:
            tmp.write(code)
            tmp.close()
            result = run_shell(f"python3 {tmp.name}", str(self.repo_root), timeout=30)
            output = result.output
            if len(output) > 5000:
                output = output[:2500] + "\n...[truncated]...\n" + output[-2500:]
            return f"{output}\n[exit {result.returncode}]"
        finally:
            import os
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    def _current_datetime(self) -> str:
        import datetime
        now = datetime.datetime.now()
        utc = datetime.datetime.now(datetime.timezone.utc)
        try:
            tz_name = datetime.datetime.now().astimezone().tzname()
        except Exception:
            tz_name = "local"
        return (
            f"Current local time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})\n"
            f"UTC time: {utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"Day: {now.strftime('%A')}"
        )

    def _search_code(self, query: str) -> str:
        results = search_index(self.repo_root, query)
        if not results:
            count, path = build_index(self.repo_root)
            results = search_index(self.repo_root, query)
            if not results:
                return f"No code index matches. Built index for {count} files."
        lines = [f"{item['path']}#chunk{item['chunk_id']}\n{item['preview']}" for item in results]
        return "\n\n".join(lines)

    # ── Web search implementations ───────────────────────────────────────

    def _web_search(self, query: str, max_results: int = 5) -> str:
        # Try DuckDuckGo, fall back to scraping if needed.
        result = self._duckduckgo_search(query, max_results)
        if result == "No results found." or "Search error" in result:
            # Fallback: try bash curl to get a quick answer
            try:
                from .shell import run_shell
                sr = run_shell(f'curl -s "https://lite.duckduckgo.com/lite/?q={query.replace(" ", "+")}" 2>&1 | head -100', str(self.repo_root), timeout=10)
                if sr.output.strip():
                    return f"Web results for '{query}':\n{sr.output[:2000]}"
            except Exception:
                pass
        return result

    def _web_fetch(self, url: str, max_chars: int = 12000) -> str:
        try:
            response = httpx.get(
                url, timeout=20.0, follow_redirects=True,
                headers={"User-Agent": "LocalCode/0.2 (coding-assistant)"},
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text" not in content_type and "json" not in content_type and "xml" not in content_type:
                return f"Non-text content type: {content_type}"
            text = response.text
            # Strip HTML tags for readability
            if "html" in content_type:
                text = self._strip_html(text)
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[truncated]"
            return text
        except Exception as exc:
            return f"Fetch failed: {exc}"

    @staticmethod
    def _strip_html(html: str) -> str:
        # Remove script and style tags entirely
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Collapse multiple newlines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _duckduckgo_search(self, query: str, max_results: int = 5) -> str:
        results: list[str] = []
        try:
            with DDGS() as ddgs:
                # Try instant answer first (for time, weather, calculations, etc.)
                try:
                    answers = list(ddgs.answers(query))
                    if answers:
                        for ans in answers[:2]:
                            text = ans.get("text", "")
                            url = ans.get("url", "")
                            if text:
                                results.append(f"[Instant Answer] {text}\n{url}")
                except Exception:
                    pass
                # Then regular search results
                for item in ddgs.text(query, max_results=max_results):
                    title = item.get("title", "")
                    body = item.get("body", "")
                    href = item.get("href", "")
                    results.append(f"{title}\n{body}\n{href}")
        except Exception as exc:
            return f"Search error: {exc}"
        return "\n\n".join(results) if results else "No results found."

    # ── Plugin & MCP ─────────────────────────────────────────────────────

    def _register_plugin_tools(self) -> None:
        """Plugin system removed — no-op."""
        pass

    @staticmethod
    def _make_mcp_handler(client: Any, tool_name: str) -> ToolHandler:
        """Build a handler bound to a specific client + tool name.

        Defined as its own method (not a lambda inside a loop) so the
        client/tool_name are captured per-tool — no late-binding bug where
        every registered MCP tool would point at the last loop iteration.
        """
        def handler(args: dict[str, Any]) -> str:
            try:
                return client.call_tool(tool_name, args or {})
            except Exception as exc:  # never let a bad MCP call crash dispatch
                return f"MCP tool {tool_name!r} call failed: {exc}"
        return handler

    def ensure_mcp_tools(self) -> None:
        """Lazily connect configured MCP servers and register their tools.

        Runs at most once per toolkit (guarded by self._mcp_loaded). Each
        connected server's tools are registered as `mcp_<server>_<tool>`
        LocalCodeTools so the model can call them like any builtin. Fully
        defensive: a bad/missing MCP server never crashes the toolkit.
        """
        if self._mcp_loaded:
            return
        self._mcp_loaded = True  # set first so we only ever attempt once
        try:
            from . import mcp as _mcp
        except Exception as exc:
            self.mcp_errors.append(f"mcp import failed: {exc}")
            return
        try:
            _count, errors = _mcp.connect_all()
            if errors:
                self.mcp_errors.extend(errors)
            for server, tools in _mcp.list_connected():
                client = _mcp.get_client(server)
                if client is None:
                    continue
                self.mcp_clients[server] = client
                for t in tools:
                    tool_name = (t.get("name") or "").strip()
                    if not tool_name:
                        continue
                    schema = t.get("inputSchema") or {"type": "object", "properties": {}}
                    if not isinstance(schema, dict):
                        schema = {"type": "object", "properties": {}}
                    description = (
                        f"[MCP {server}] " + (t.get("description") or "")
                    ).strip()[:1000]
                    self._register(LocalCodeTool(
                        name=f"mcp_{server}_{tool_name}",
                        description=description,
                        parameters=schema,
                        handler=self._make_mcp_handler(client, tool_name),
                    ))
        except Exception as exc:  # connect_all/list shouldn't raise, but be safe
            self.mcp_errors.append(f"mcp wiring failed: {exc}")

    # ── Public API ───────────────────────────────────────────────────────

    # Core tools that small models can handle without choking
    # Core tools for small models
    CORE_TOOLS = {
        "read_file", "write_file", "edit_file",
        "bash", "web_search", "current_datetime",
        "tool_search",
    }

    def schemas(self, compact: bool = False, minimal: bool = False) -> list[dict[str, Any]]:
        """Get tool schemas.

        compact: only core tools
        minimal: ultra-short descriptions for small models (saves ~500 tokens)
        """
        self.ensure_mcp_tools()
        if compact:
            tools = [tool for tool in self.tools.values() if tool.name in self.CORE_TOOLS]
        else:
            tools = list(self.tools.values())
        if minimal:
            return [self._minimal_schema(tool) for tool in tools]
        return [tool.as_schema() for tool in tools]

    @staticmethod
    def _minimal_schema(tool: LocalCodeTool) -> dict[str, Any]:
        """Ultra-compact schema — one-line description, minimal params."""
        # Strip description to first sentence
        desc = tool.description.split(".")[0].strip() + "."
        # Keep only required params
        params = tool.parameters.copy()
        props = params.get("properties", {})
        required = params.get("required", [])
        # Only include required properties
        if required:
            props = {k: {"type": v.get("type", "string")} for k, v in props.items() if k in required}
        else:
            props = {k: {"type": v.get("type", "string")} for k, v in list(props.items())[:2]}
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    def list_tool_names(self) -> list[str]:
        self.ensure_mcp_tools()
        return sorted(self.tools.keys())

    def summarize_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[str]:
        summaries: list[str] = []
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            raw_args = function.get("arguments", {})
            if isinstance(raw_args, str):
                args_text = raw_args
            else:
                args_text = json.dumps(raw_args)
            if len(args_text) > 180:
                args_text = args_text[:180] + "..."
            summaries.append(f"{name}({args_text})")
        return summaries

    # Tools safe to run concurrently (read-only, no side effects)
    READ_ONLY_TOOLS = {
        "read_file", "grep", "glob", "list_files", "search_code",
        "git_status", "git_diff", "git_log", "current_datetime",
        "web_search", "web_fetch",
    }

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Execute tool calls — read-only tools run concurrently, writes run serially."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        self.ensure_mcp_tools()

        if len(tool_calls) <= 1:
            return self._execute_serial(tool_calls)

        # Partition into read-only (concurrent) and write (serial) batches
        read_calls = []
        write_calls = []
        for call in tool_calls:
            name = call.get("function", {}).get("name", "")
            if name in self.READ_ONLY_TOOLS:
                read_calls.append(call)
            else:
                write_calls.append(call)

        outputs: list[dict[str, str]] = []

        # Run read-only tools concurrently
        if read_calls:
            with ThreadPoolExecutor(max_workers=min(4, len(read_calls))) as pool:
                futures = {}
                for i, call in enumerate(read_calls):
                    futures[pool.submit(self._execute_one, call)] = i
                results = [None] * len(read_calls)
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        results[idx] = {"role": "tool", "content": f"Tool error: {exc}"}
                        # Abort siblings on error — cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                outputs.extend(r for r in results if r is not None)

        # Run write tools serially
        outputs.extend(self._execute_serial(write_calls))
        return outputs

    def _execute_one(self, call: dict[str, Any]) -> dict[str, str]:
        """Execute a single tool call with input validation."""
        function = call.get("function", {})
        # Strip whitespace from the tool name. Quantized models (Qwen 3.6
        # IQ2_M in particular) sometimes emit names with a trailing space
        # like 'list_files ' — without this strip, every such call would
        # fall through to "Unknown tool" with the trailing space visible
        # in the error, which the user kept reporting.
        name = (function.get("name", "") or "").strip()
        raw_args = function.get("arguments", {})
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        tool = self.tools.get(name)
        if tool is None:
            return {"role": "tool", "content": f"Unknown tool: '{name}'. Available: {', '.join(sorted(self.tools.keys())[:8])}…"}
        # Validate required params
        required = tool.parameters.get("required", [])
        missing = [r for r in required if r not in (args or {})]
        if missing:
            return {"role": "tool", "content": f"Missing required args for {name}: {missing}"}
        try:
            content = tool.handler(args or {})
        except Exception as exc:
            content = f"Tool error for {name}: {exc}"
        return {"role": "tool", "content": content}

    def _execute_serial(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Execute tool calls one at a time."""
        return [self._execute_one(call) for call in tool_calls]

    def close(self) -> None:
        for client in self.mcp_clients.values():
            client.close()

    def search_status(self) -> tuple[str, str]:
        return "duckduckgo", "configured"

    def diagnostics(self) -> list[str]:
        rows: list[str] = []
        rows.extend(f"plugin error: {item}" for item in self.plugin_errors)
        if not self._mcp_loaded:
            rows.append("mcp: deferred until needed")
            return rows
        rows.extend(f"mcp error: {item}" for item in self.mcp_errors)
        for name, client in self.mcp_clients.items():
            ok, detail = client.health()
            rows.append(f"mcp {name}: {'ok' if ok else 'bad'} ({detail})")
        return rows
