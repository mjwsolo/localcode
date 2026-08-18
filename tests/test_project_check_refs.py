"""Regression: the build-completion typecheck gate must use `tsc -b` for TS
PROJECT-REFERENCE scaffolds (Vite/React/Vue), not `tsc --noEmit`.

A modern scaffold's root tsconfig.json is a solution file (`"references": [...]`,
no `include`). `tsc --noEmit` on it type-checks NOTHING and reports a false
clean — the exact hole that let an Anki build with 3 real `tsc` errors complete
as "verified". Only `tsc -b` (what `npm run build` runs) surfaces those errors.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from localcode.tools.project_check import _detect_commands


def _scaffold(tmp: Path, tsconfig: dict) -> None:
    (tmp / "package.json").write_text(json.dumps({"name": "x", "scripts": {}}))
    (tmp / "tsconfig.json").write_text(json.dumps(tsconfig))
    binp = tmp / "node_modules" / ".bin"
    binp.mkdir(parents=True)
    (binp / "tsc").write_text("#!/bin/sh\nexit 0\n")
    os.chmod(binp / "tsc", 0o755)


def test_project_references_use_tsc_b(tmp_path: Path):
    _scaffold(tmp_path, {"references": [{"path": "./tsconfig.app.json"}]})
    labels = [label for label, _argv, _cwd in _detect_commands(str(tmp_path))]
    assert "tsc -b" in labels
    assert "tsc --noEmit" not in labels
    # and the argv is build mode
    argv = next(a for lbl, a, _ in _detect_commands(str(tmp_path)) if lbl == "tsc -b")
    assert argv[1] == "-b"


def test_plain_tsconfig_uses_noemit(tmp_path: Path):
    _scaffold(tmp_path, {"compilerOptions": {"strict": True}, "include": ["src"]})
    labels = [label for label, _argv, _cwd in _detect_commands(str(tmp_path))]
    assert "tsc --noEmit" in labels
    assert "tsc -b" not in labels
