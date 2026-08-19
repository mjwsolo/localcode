"""inspect_symbol — return an installed library's REAL signature/types.

A weak local model tends to GUESS a third-party API (e.g. call `fsrs.repeat()`
and index `[1]`, ignoring the rating arg) instead of reading the installed
package. This tool resolves a package name to its installed type declarations
(`.d.ts` for JS/TS, `.pyi`/source for Python) and returns the actual signature
for a symbol — so the model calls the real API, not a hallucinated one.

Filesystem-only by design: no language server, no code execution (safe to run
unattended, and cheap enough for a 16 GB Mac running a big quantized model).
This is the "LSP hover, as a tool you call on demand" mechanism.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "inspect_symbol",
        "description": (
            "Look up the REAL signature/types of a library symbol from its INSTALLED "
            "source, so you call the actual API instead of guessing. Give the package "
            "name and (optionally) the symbol. Returns the declaration from "
            "node_modules `.d.ts` (JS/TS) or site-packages `.pyi`/source (Python). "
            "Use it BEFORE calling any third-party API you are not 100% sure of "
            "(argument order, return shape, method names)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Package/module name, e.g. 'fsrs', 'react', 'numpy'.",
                },
                "symbol": {
                    "type": "string",
                    "description": (
                        "Optional: the function/class/type/method to look up, e.g. "
                        "'repeat'. Omit to list the module's exported names."
                    ),
                },
            },
            "required": ["module"],
        },
    },
}

_MAX_CHARS = 3500
_DECL_RE_TMPL = (
    r"\b(?:export\s+)?(?:declare\s+)?(?:default\s+)?"
    r"(?:async\s+)?(?:function|const|let|var|class|interface|type|enum|namespace|abstract\s+class)\s+{sym}\b"
)


def execute(ctx: ToolContext, args: dict) -> str:
    module = str(args.get("module") or "").strip()
    if not module:
        return "Error: 'module' is required (the package/module name)."
    symbol = str(args.get("symbol") or "").strip()
    repo = ctx.repo

    ts = _inspect_ts(repo, module, symbol)
    if ts:
        return ts[:_MAX_CHARS]
    py = _inspect_python(repo, module, symbol)
    if py:
        return py[:_MAX_CHARS]
    return (
        f"No installed types found for '{module}'. If it's a dependency, make sure it's "
        f"installed (npm install / pip install) then retry. You can also read its source "
        f"directly (node_modules/{module}/ for JS/TS, the site-packages dir for Python)."
    )


# ── TypeScript / JavaScript ───────────────────────────────────────────────

def _find_node_modules_pkg(repo: Path, module: str) -> Path | None:
    """Locate node_modules/<module>, checking the repo root and one level of
    subdirectories (scaffolds often live in a subfolder)."""
    candidates = [repo / "node_modules" / module]
    try:
        for child in sorted(repo.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                candidates.append(child / "node_modules" / module)
    except OSError:
        pass
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _dts_files(pkg_dir: Path, repo: Path, module: str) -> list[Path]:
    files: list[Path] = []
    # 1) the package's declared entry types
    pj = pkg_dir / "package.json"
    if pj.is_file():
        try:
            data = json.loads(pj.read_text(errors="replace"))
            entry = data.get("types") or data.get("typings")
            if entry:
                p = (pkg_dir / str(entry)).resolve()
                if p.is_file():
                    files.append(p)
        except Exception:
            pass
    # 2) @types/<module> (DefinitelyTyped)
    nm = pkg_dir.parent
    at = nm / f"@types" / module / "index.d.ts"
    if at.is_file():
        files.append(at)
    # 3) fallback: bounded glob of the package's own .d.ts files
    if not files:
        try:
            files.extend(sorted(pkg_dir.rglob("*.d.ts"))[:12])
        except OSError:
            pass
    # de-dup, preserve order
    seen: set[str] = set()
    out: list[Path] = []
    for f in files:
        k = str(f)
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def _inspect_ts(repo: Path, module: str, symbol: str) -> str | None:
    pkg = _find_node_modules_pkg(repo, module)
    if not pkg:
        return None
    files = _dts_files(pkg, repo, module)
    if not files:
        return None
    if not symbol:
        # List exported names from the entry declaration file.
        head = files[0]
        text = _read(head)
        exports = sorted(set(re.findall(r"\bexport\s+(?:declare\s+)?(?:default\s+)?(?:async\s+)?(?:function|const|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)", text)))
        rel = _rel(head, repo)
        if exports:
            return f"[{module}] exports (from {rel}):\n" + ", ".join(exports[:60])
        return f"[{module}] type entry: {rel}\n\n" + text[:_MAX_CHARS]
    # Extract declaration blocks for the symbol across the type files.
    blocks = _extract_blocks(files, symbol, repo)
    if blocks:
        return f"[{module}.{symbol}] real signature(s) from installed types:\n\n" + blocks
    return None


# ── Python ────────────────────────────────────────────────────────────────

def _find_python_pkg(repo: Path, module: str) -> list[Path]:
    """Find installed <module> source/stub files under the project's venv or a
    site-packages dir. Best-effort, bounded."""
    hits: list[Path] = []
    top = module.split(".")[0]
    roots: list[Path] = []
    try:
        for sp in list(repo.rglob("site-packages"))[:4]:
            roots.append(sp)
    except OSError:
        pass
    for sp in roots:
        for name in (f"{top}.pyi", f"{top}.py", f"{top}/__init__.pyi", f"{top}/__init__.py"):
            p = sp / name
            if p.is_file():
                hits.append(p)
    return hits[:4]


def _inspect_python(repo: Path, module: str, symbol: str) -> str | None:
    files = _find_python_pkg(repo, module)
    if not files:
        return None
    if not symbol:
        text = _read(files[0])
        names = sorted(set(re.findall(r"^(?:def|class)\s+([A-Za-z_]\w*)", text, re.M)))
        if names:
            return f"[{module}] top-level defs (from {_rel(files[0], repo)}):\n" + ", ".join(names[:60])
        return f"[{module}] {_rel(files[0], repo)}\n\n" + text[:_MAX_CHARS]
    blocks = _extract_py_blocks(files, symbol, repo)
    if blocks:
        return f"[{module}.{symbol}] real signature(s) from installed source:\n\n" + blocks
    return None


# ── shared extraction ─────────────────────────────────────────────────────

def _extract_blocks(files: list[Path], symbol: str, repo: Path) -> str:
    """Return declaration windows for `symbol` from TS declaration files."""
    decl_re = re.compile(_DECL_RE_TMPL.format(sym=re.escape(symbol)))
    # also catch a class METHOD or property signature: `symbol(...)` / `symbol:`
    method_re = re.compile(rf"^\s*(?:readonly\s+|public\s+|static\s+|abstract\s+)*{re.escape(symbol)}\s*[<(:]")
    out: list[str] = []
    count = 0
    for f in files:
        lines = _read(f).splitlines()
        for i, line in enumerate(lines):
            if decl_re.search(line) or method_re.search(line):
                window = _window(lines, i)
                out.append(f"// {_rel(f, repo)}:{i + 1}\n" + window)
                count += 1
                if count >= 6:
                    return "\n\n".join(out)
    return "\n\n".join(out)


def _extract_py_blocks(files: list[Path], symbol: str, repo: Path) -> str:
    decl_re = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b")
    out: list[str] = []
    count = 0
    for f in files:
        lines = _read(f).splitlines()
        for i, line in enumerate(lines):
            if decl_re.search(line):
                window = _window(lines, i, stop_on_dedent=True)
                out.append(f"# {_rel(f, repo)}:{i + 1}\n" + window)
                count += 1
                if count >= 6:
                    return "\n\n".join(out)
    return "\n\n".join(out)


def _window(lines: list[str], start: int, stop_on_dedent: bool = False, max_lines: int = 14) -> str:
    """Capture a declaration block from `start`: until a blank line / a line that
    closes the statement (`;`, `}`) / a dedent to the declaration's indent."""
    out = [lines[start]]
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    depth = lines[start].count("{") - lines[start].count("}")
    for j in range(start + 1, min(len(lines), start + max_lines)):
        ln = lines[j]
        if stop_on_dedent:
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= base_indent:
                break
            out.append(ln)
            continue
        out.append(ln)
        depth += ln.count("{") - ln.count("}")
        if depth <= 0 and (ln.rstrip().endswith(";") or ln.rstrip().endswith("}") or not ln.strip()):
            break
    return "\n".join(out).rstrip()


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def _rel(p: Path, repo: Path) -> str:
    try:
        return str(p.relative_to(repo))
    except ValueError:
        return str(p)


def is_concurrency_safe(args: dict) -> bool:
    return True
