"""inspect_symbol: return a library's REAL signature from installed types, so a
weak model calls the actual API instead of guessing it (the fsrs.repeat()[1]
failure class)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from localcode.tools import AVAILABLE_NAMES, dispatch
from localcode.tools import inspect_symbol


def _ctx(repo: Path) -> SimpleNamespace:
    return SimpleNamespace(repo=repo, app=SimpleNamespace(repo_root=repo))


def _npm_pkg(repo: Path, name: str, dts: str, types_entry: str = "index.d.ts") -> None:
    pkg = repo / "node_modules" / name
    pkg.mkdir(parents=True)
    (pkg / "package.json").write_text(json.dumps({"name": name, "types": types_entry}))
    (pkg / types_entry).write_text(dts)


def test_registered():
    assert "inspect_symbol" in AVAILABLE_NAMES


def test_returns_real_ts_signature(tmp_path: Path):
    _npm_pkg(tmp_path, "fsrs", (
        "export declare enum Rating { Again = 1, Hard = 2, Good = 3, Easy = 4 }\n"
        "export declare class FSRS {\n"
        "  repeat(card: Card, now: Date): Record<Rating, RecordLogItem>;\n"
        "  next(card: Card, now: Date, grade: Rating): RecordLogItem;\n"
        "}\n"
    ))
    out = dispatch("inspect_symbol", _ctx(tmp_path), {"module": "fsrs", "symbol": "repeat"})
    # the model must SEE that repeat returns a record keyed by Rating (index by
    # the rating), not a positional array — the whole point of the tool.
    assert "repeat(card: Card, now: Date): Record<Rating, RecordLogItem>" in out
    assert "guess" not in out.lower() or "installed types" in out.lower()


def test_lists_exports_without_symbol(tmp_path: Path):
    _npm_pkg(tmp_path, "fsrs",
             "export declare class FSRS {}\nexport declare enum Rating {}\n")
    out = dispatch("inspect_symbol", _ctx(tmp_path), {"module": "fsrs"})
    assert "FSRS" in out and "Rating" in out


def test_finds_pkg_in_subdirectory(tmp_path: Path):
    # scaffolds live in a subfolder; the tool must find node_modules there.
    (tmp_path / "app").mkdir()
    _npm_pkg(tmp_path / "app", "dexie",
             "export declare class Dexie {\n  version(n: number): Version;\n}\n")
    out = dispatch("inspect_symbol", _ctx(tmp_path), {"module": "dexie", "symbol": "version"})
    assert "version(n: number)" in out


def test_not_installed_is_actionable(tmp_path: Path):
    out = dispatch("inspect_symbol", _ctx(tmp_path), {"module": "nope", "symbol": "x"})
    assert "No installed types" in out or "npm install" in out


def test_missing_module_arg_is_rejected(tmp_path: Path):
    # dispatch enforces the schema's required fields.
    out = dispatch("inspect_symbol", _ctx(tmp_path), {})
    assert "module" in out.lower()


def test_python_signature(tmp_path: Path):
    sp = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    (sp / "widget.py").write_text("def render(node, depth: int = 0) -> str:\n    return ''\n")
    out = dispatch("inspect_symbol", _ctx(tmp_path), {"module": "widget", "symbol": "render"})
    assert "def render(node, depth: int = 0) -> str" in out
