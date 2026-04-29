"""Offline code diagnostics for post-write verification.

Runs `ruff` → `pyflakes` → `python -m py_compile` as a fallback chain
on a Python file after the model writes it, surfacing syntax errors
and obvious bugs before the model proceeds. Called from `app.py:1219`
in the "Updated files:" summary.

## History

This module used to be ~650 LoC implementing a real LSP client
(spawning pyright / typescript-language-server / gopls / rust-analyzer
via JSON-RPC), plus jedi-based goto/references/completions, plus an
`EnhancedChecker` abstraction layer. **None of that was wired to the
live app** — the only external caller was `app.py:1219` calling the
module-level `get_diagnostics(path)` function which uses subprocess.

Trimmed to offline-only during the T0.9 dead-code sweep. If the real
LSP layer becomes interesting again (cross-file symbol rename,
incremental type checks), reintroduce it as a separate module —
don't resurrect the deleted one; the design had accumulated enough
half-finished paths that it was easier to start fresh.

## Why offline-only is fine for our use case

LocalCode's post-write check is narrow: "did the model produce a
syntactically valid Python file?" `ruff check --output-format=json`
answers that in ~50 ms per file, no daemon, no process lifecycle,
no incremental-state bookkeeping. The cost of the real LSP layer
(process management, JSON-RPC, initialisation handshakes) only pays
off for interactive use cases like autocomplete or live hover —
neither of which the agent loop consumes.

If we ever wire LSP for model feedback (e.g. "after writing foo.py,
hand the diagnostics back to the model so it can fix errors in the
next round"), offline tools still give us everything needed for
Python. Other languages would need real LSP, but we'd scope that
per-language when we get there.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


__all__ = ["Diagnostic", "get_diagnostics"]


@dataclass
class Diagnostic:
    """One finding from a linter or syntax check.

    `severity` is a plain string ("error" | "warning" | "info") so
    callers can colour-map without importing an enum. `source` is
    the tool that produced this finding (ruff / pyflakes /
    py_compile) — useful for telling the user WHY a given finding
    showed up and not another.
    """
    file: str
    line: int
    severity: str   # "error" | "warning" | "info"
    message: str
    source: str = ""

    def __str__(self) -> str:
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(self.severity, "?")
        return f"  {icon} {self.file}:{self.line} {self.message}"


def get_diagnostics(file_path: Path) -> list[Diagnostic]:
    """Run offline diagnostics on a Python file.

    Tries tools in order of precision: ruff (fast, lots of rules) →
    pyflakes (simpler, available on more systems) → py_compile
    (syntax-only, always-available fallback). Returns the first
    non-empty result; no tool runs a second time if an earlier one
    produced findings.

    Returns an empty list for non-.py files (silently — the caller
    at app.py:1218 already filters by extension before calling).

    All subprocess calls have a 10-second timeout; any tool failure
    (binary not found, timeout, malformed JSON) is swallowed and the
    function falls through to the next tool. Worst case: no
    diagnostics returned, caller displays "No issues found" and the
    model proceeds.
    """
    diagnostics: list[Diagnostic] = []
    rel = file_path.name

    if file_path.suffix != ".py":
        return diagnostics

    # Ruff — fast Python linter. Preferred when available because it
    # covers flake8 + pyflakes + pep8 rules in one binary.
    if shutil.which("ruff"):
        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", str(file_path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip():
                for item in json.loads(result.stdout):
                    diagnostics.append(Diagnostic(
                        file=rel,
                        line=item.get("location", {}).get("row", 0),
                        severity="warning",
                        message=f"{item.get('code', '')}: {item.get('message', '')}",
                        source="ruff",
                    ))
            return diagnostics
        except Exception:
            pass

    # Pyflakes — older but widely installed alternative.
    if shutil.which("pyflakes"):
        try:
            result = subprocess.run(
                ["pyflakes", str(file_path)],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    try:
                        lineno = int(parts[1])
                    except ValueError:
                        lineno = 0
                    diagnostics.append(Diagnostic(
                        file=rel, line=lineno, severity="warning",
                        message=parts[2].strip(), source="pyflakes",
                    ))
            return diagnostics
        except Exception:
            pass

    # py_compile — syntax-only, always-available final fallback.
    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(file_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "syntax error"
            diagnostics.append(Diagnostic(
                file=rel, line=0, severity="error", message=err, source="py_compile",
            ))
    except Exception:
        pass

    return diagnostics
