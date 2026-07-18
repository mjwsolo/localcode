"""Deterministic semantic navigation without a resident language server.

Python uses its AST for definitions/symbols. References use identifier-boundary
search, which is deliberately conservative and works for every text language.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "code_navigation",
        "description": "Find code symbols, definitions, or references deterministically. Prefer this over repeated grep/read guesses.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["symbols", "definition", "references"]},
                "symbol": {"type": "string", "description": "Required for definition/references."},
                "path": {"type": "string", "description": "Optional repository-relative file or directory."},
                "max_results": {"type": "integer", "default": 50},
            },
            "required": ["action"],
        },
    },
}

_SKIP = {".git", ".localcode", "node_modules", ".venv", "venv", "dist", "build", "__pycache__"}

def _files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return [p for p in root.rglob("*") if p.is_file() and not any(x in _SKIP for x in p.parts)]

def _python_symbols(path: Path) -> list[tuple[str, int, str]]:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (OSError, SyntaxError, UnicodeError):
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append((node.name, node.lineno, "class" if isinstance(node, ast.ClassDef) else "function"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    found.append((target.id, node.lineno, "variable"))
    return sorted(found, key=lambda item: item[1])

def execute(ctx: ToolContext, args: dict) -> str:
    action = str(args.get("action", ""))
    symbol = str(args.get("symbol", "")).strip()
    limit = max(1, min(int(args.get("max_results", 50)), 500))
    target = ctx.resolve_path(str(args.get("path", ""))).resolve()
    try:
        target.relative_to(ctx.repo.resolve())
    except ValueError:
        return "Error: path must stay inside the repository."
    if not target.exists():
        return f"Error: path not found: {args.get('path', '')}"
    if action in {"definition", "references"} and not symbol:
        return f"Error: symbol is required for {action}."

    results: list[str] = []
    boundary = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])") if symbol else None
    for path in _files(target):
        rel = path.relative_to(ctx.repo)
        if path.suffix == ".py" and action in {"symbols", "definition"}:
            for name, line, kind in _python_symbols(path):
                if action == "symbols" or name == symbol:
                    results.append(f"{rel}:{line}: {kind} {name}")
        elif action == "symbols":
            continue
        if action == "references":
            try:
                for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if boundary and boundary.search(line):
                        results.append(f"{rel}:{line_no}: {line.strip()[:240]}")
                        if len(results) >= limit:
                            break
            except OSError:
                pass
        if len(results) >= limit:
            break
    if not results:
        return f"No {action} results found" + (f" for {symbol!r}." if symbol else ".")
    clipped = results[:limit]
    suffix = f"\n[limited to {limit} results]" if len(results) > limit else ""
    return "\n".join(clipped) + suffix

def is_concurrency_safe(args: dict) -> bool:
    return True
