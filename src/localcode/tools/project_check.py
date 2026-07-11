"""Tier-2 verification: run the project's OWN typecheck/lint once, return errors.

The big agents catch semantic errors with a persistent language server (LSP).
opencode's own docs say that for many projects it's better to run the project's
diagnostic CLI directly — LSPs "get out of sync, use significant memory … and
slow down agent workflows." That's decisive for localcode: a 16 GB Mac already
running a large local model can't spare an always-on tsserver/pyright.

So instead we run the project's real typecheck ONCE, when the model is about to
finish an unverified build, and feed the concrete errors back deterministically
(the model can't skip it, and gets ground truth like "line 37: 'getCardsForCard'
does not exist" — the semantic errors a per-write syntax check can't catch).

Light: one bounded subprocess at completion, not per-edit. Returns None when the
project is clean or no checker is available.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

_TIMEOUT = 60.0
_MAX_ERR_CHARS = 2500
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_project_check(repo_root: str) -> str | None:
    """Run the first available typecheck/lint for the project; return a bounded
    error summary, or None if clean / nothing to run."""
    env = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"}
    for label, argv, cwd in _detect_commands(repo_root):
        try:
            r = subprocess.run(
                argv, cwd=cwd, capture_output=True, text=True,
                timeout=_TIMEOUT, env=env,
            )
        except Exception:
            continue
        if r.returncode != 0:
            out = _ANSI.sub("", ((r.stdout or "") + "\n" + (r.stderr or "")).strip())
            # Keep only the most useful lines, capped, so we don't flood a
            # small model's context with a wall of output.
            lines = [l for l in out.splitlines() if l.strip()][:40]
            if lines:
                return f"[{label}] reported errors:\n" + "\n".join(lines)[:_MAX_ERR_CHARS]
    return None


def _detect_commands(repo_root: str) -> list[tuple[str, list[str], str]]:
    """Ordered (label, argv, cwd) checkers to try. First that runs wins."""
    cmds: list[tuple[str, list[str], str]] = []

    # ── JS / TS ── prefer a project "typecheck" script, else tsc --noEmit.
    pj_dir = _nearest_with(repo_root, "package.json")
    if pj_dir:
        node_modules = os.path.join(pj_dir, "node_modules")
        binp = os.path.join(node_modules, ".bin")
        scripts = {}
        try:
            scripts = (json.load(open(os.path.join(pj_dir, "package.json"))) or {}).get("scripts", {})
        except Exception:
            scripts = {}
        # Only run node-based checks once deps are installed (else every import
        # is a false "cannot find module" error that would derail the model).
        # NOTE: tsc/typecheck only covers the files the project's tsconfig
        # includes (typically `src/`). A module written OUTSIDE that root (e.g.
        # accidentally at the repo root) is never type-checked here, so the
        # Tier-1 per-write syntax_check is the only gate it passes through. If
        # this proves a recurring miss, widen coverage (e.g. an explicit tsc
        # over stray *.ts outside `include`) rather than assuming src/-only.
        if os.path.isdir(node_modules):
            if "typecheck" in scripts:
                cmds.append(("npm run typecheck", ["npm", "run", "--silent", "typecheck"], pj_dir))
            elif os.path.exists(os.path.join(pj_dir, "tsconfig.json")) and os.path.exists(os.path.join(binp, "tsc")):
                cmds.append(("tsc --noEmit", [os.path.join(binp, "tsc"), "--noEmit", "--pretty", "false"], pj_dir))
            elif "lint" in scripts:
                cmds.append(("npm run lint", ["npm", "run", "--silent", "lint"], pj_dir))

    # ── Python ── ruff for REAL errors only (E9 syntax + F pyflakes: undefined
    # names, bad imports) — not style noise. Falls back to compileall (syntax).
    if _has_ext(repo_root, ".py"):
        if shutil.which("ruff"):
            cmds.append(("ruff", ["ruff", "check", "--select", "E9,F",
                                  "--no-cache", "--output-format", "concise", "."], repo_root))
        elif shutil.which("python3") or shutil.which("python"):
            py = shutil.which("python3") or shutil.which("python")
            cmds.append(("python -m compileall", [py, "-m", "compileall", "-q", "."], repo_root))

    # ── Go ── the compiler catches type/undefined errors (like tsc). go.mod.
    go_dir = _nearest_with(repo_root, "go.mod")
    if go_dir and shutil.which("go"):
        cmds.append(("go build", ["go", "build", "./..."], go_dir))

    # ── Rust ── cargo check is the type-checker (fast, no codegen). Cargo.toml.
    rust_dir = _nearest_with(repo_root, "Cargo.toml")
    if rust_dir and shutil.which("cargo"):
        cmds.append(("cargo check", ["cargo", "check", "--message-format", "short"], rust_dir))

    # (Interpreted langs — Ruby/PHP/shell — have no whole-project type-check;
    # their per-file syntax linters run in the Tier-1 syntax_check on each write.)
    return cmds


def _nearest_with(root: str, filename: str) -> str | None:
    """The shallowest directory under root that contains `filename`."""
    root = os.path.abspath(root)
    if os.path.exists(os.path.join(root, filename)):
        return root
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "dist", "build", ".venv")]
        if filename in files:
            return base
    return None


def _has_ext(root: str, ext: str) -> bool:
    for base, dirs, files in os.walk(os.path.abspath(root)):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "dist", "build", ".venv")]
        if any(f.endswith(ext) for f in files):
            return True
    return False
