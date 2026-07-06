"""One-shot syntax validation for files the agent writes.

Poor-man's LSP. The big agents (Claude Code, opencode) run a persistent
language server and feed its diagnostics back after each edit; codex uses
verified structured patches. A persistent LSP is too heavy for localcode's
target — a 16 GB Mac already running a large local model. So instead we run a
FAST, ONE-SHOT syntax check after each write and, if it fails, append the exact
error to the tool result.

Why this matters for a LOCAL model specifically: a small quantized model drifts
into structural typos (`useState(0]`, an unterminated string, a missing bracket,
`getCardsForCard`). With no verification the model has to NOTICE its own typo —
which weak models are bad at — so it re-reads the broken file, the broken code
enters its context, and it reproduces the error (self-conditioning loop). A
deterministic check gives it GROUND TRUTH ("line 37: unexpected token") so it
fixes the exact line instead of guessing.

Best-effort and non-blocking: the write ALWAYS succeeds; this only appends a
warning. Any checker failure/absence returns None (no false alarms). Disable
with LOCALCODE_SYNTAX_CHECK=0.
"""
from __future__ import annotations

import json
import os
import subprocess

_TIMEOUT = 5.0


def check_syntax(path: str, content: str) -> str | None:
    """Return a one-line syntax-error string, or None if OK / not checkable.

    In-process for JSON/Python (instant, zero deps); a fast one-shot subprocess
    for JS (node --check) and TS/JSX (esbuild, when the project has it).
    """
    if os.environ.get("LOCALCODE_SYNTAX_CHECK") == "0":
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            return _check_json(content)
        if ext in (".py", ".pyi"):
            return _check_python(content, path)
        if ext in (".js", ".mjs", ".cjs"):
            return _check_node(path)
        if ext in (".ts", ".tsx", ".jsx", ".mts", ".cts"):
            return _check_esbuild(path)
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


def _check_node(path: str) -> str | None:
    import shutil
    if not shutil.which("node"):
        return None
    r = subprocess.run(
        ["node", "--check", path],
        capture_output=True, text=True, timeout=_TIMEOUT,
    )
    if r.returncode != 0:
        return _first_error_line(r.stderr)
    return None


def _find_bin(start: str, name: str) -> str | None:
    """Walk up from the file's dir to find node_modules/.bin/<name>."""
    d = os.path.dirname(os.path.abspath(start))
    for _ in range(25):
        cand = os.path.join(d, "node_modules", ".bin", name)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _check_esbuild(path: str) -> str | None:
    # esbuild parses TS/TSX/JSX and reports syntax errors instantly. Only used
    # when the project has it (Vite/most JS projects do post-install); if not
    # present we skip rather than risk a false alarm from a wrong parser.
    esbuild = _find_bin(path, "esbuild")
    if not esbuild:
        return None
    loader = "tsx" if path.endswith((".tsx", ".jsx")) else "ts"
    r = subprocess.run(
        [esbuild, path, f"--loader={loader}", "--log-level=error"],
        capture_output=True, text=True, timeout=_TIMEOUT,
    )
    if r.returncode != 0 and r.stderr.strip():
        return _first_error_line(r.stderr)
    return None


def _first_error_line(stderr: str) -> str | None:
    """Pull the human-meaningful error out of a node/esbuild stderr dump.

    Prefer the actual `SyntaxError: …` / `error: …` message, skipping node's
    internal loader/stack frames (`node:internal/…`, `    at …`).
    """
    lines = [l.rstrip() for l in stderr.splitlines()]
    # First choice: the real error message line.
    for s in lines:
        t = s.strip()
        if t.startswith(("SyntaxError", "TypeError", "ReferenceError")) or " error:" in t.lower() or t.lower().startswith("error"):
            return t[:200]
    # Fallback: the "file:line" location node prints first (skip internals/stack).
    for s in lines:
        t = s.strip()
        if not t or t.startswith(("node:internal", "at ", "^")):
            continue
        return t[:200]
    return None
