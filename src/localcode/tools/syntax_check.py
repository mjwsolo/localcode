"""One-shot syntax validation for files the agent writes — in-process, offline.

Tier-1 verification. The big agents use a persistent language server (LSP);
opencode's own docs say that's often worse than running diagnostics directly
(LSPs "get out of sync, use significant memory, slow down agent workflows") —
decisive for a 16 GB Mac already running a large local model.

Crucially this must work with NO tools the user has to install and NO network:
it's bundled. We parse with tree-sitter (grammars compiled into the wheel — one
~4.5 MB native extension covering 100+ languages) and flag any syntax error
in-process. So when a small quantized model drifts into a structural typo
(`useState(0]`, an unterminated string, a missing bracket), the write tool
appends the exact error and line, and the model fixes THAT line instead of
re-reading the broken file and reproducing the error (the self-conditioning
loop). Python/JSON keep their native checkers for sharper messages.

Semantic/type errors (wrong names, missing imports) are caught separately by
project_check at completion (tsc/ruff/go/cargo). Disable with
LOCALCODE_SYNTAX_CHECK=0.
"""
from __future__ import annotations

import json
import os
import subprocess

_TIMEOUT = 5.0

# File extension → tree-sitter language name (all bundled in the language pack).
_TS_LANG = {
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "tsx", ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".lua": "lua",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".java": "java", ".kt": "kotlin",
    ".swift": "swift", ".cs": "csharp", ".scala": "scala",
    ".css": "css", ".scss": "scss", ".html": "html", ".xml": "xml",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".sql": "sql",
}


def check_syntax(path: str, content: str) -> str | None:
    """Return a one-line syntax-error string, or None if OK / not checkable."""
    if os.environ.get("LOCALCODE_SYNTAX_CHECK") == "0":
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        # Native checkers give the sharpest messages; use them where free.
        if ext == ".json":
            return _check_json(content)
        if ext in (".py", ".pyi"):
            return _check_python(content, path)
        lang = _TS_LANG.get(ext)
        if lang:
            return _check_treesitter(content, lang)
    except Exception:
        return None
    return None


def _check_json(content: str) -> str | None:
    try:
        json.loads(content)
        return None
    except json.JSONDecodeError as e:
        return f"JSON syntax error at line {e.lineno} col {e.colno}: {e.msg}"


def _check_python(content: str, path: str) -> str | None:
    try:
        compile(content, path, "exec")
        return None
    except SyntaxError as e:
        loc = f"line {e.lineno}" + (f" col {e.offset}" if e.offset else "")
        return f"Python syntax error at {loc}: {e.msg}"


def _check_treesitter(content: str, lang: str) -> str | None:
    """Parse with a bundled tree-sitter grammar; report the first syntax error.

    No external tools, no network — the grammars are compiled into the wheel.
    Returns None if tree-sitter isn't available (graceful) or the file parses.
    """
    try:
        from tree_sitter_language_pack import get_language
        from tree_sitter import Parser
    except Exception:
        return None
    try:
        parser = Parser(get_language(lang))
        tree = parser.parse(content.encode("utf-8"))
    except Exception:
        return None
    root = tree.root_node
    if not root.has_error:
        return None
    node = _first_error_node(root)
    if node is None:
        return f"syntax error detected in this {lang} file (unbalanced brackets/quotes or a stray token)"
    line = node.start_point[0] + 1
    col = node.start_point[1] + 1
    snippet = ""
    try:
        src_line = content.splitlines()[node.start_point[0]].strip()
        snippet = f" — near: {src_line[:80]}"
    except Exception:
        snippet = ""
    kind = "missing token" if node.is_missing else "unexpected token"
    return f"syntax error (line {line} col {col}): {kind}{snippet}"


def _first_error_node(root):
    """DFS for the first ERROR or MISSING node (the actual syntax fault)."""
    stack = [root]
    while stack:
        n = stack.pop()
        if n.is_error or n.is_missing:
            return n
        # push children in order so we return the earliest error
        stack.extend(reversed(n.children))
    return None
