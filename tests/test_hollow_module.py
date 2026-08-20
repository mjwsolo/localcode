"""A module that is created, imported, and left empty must be caught.

Reproduces the Anki-clone run that shipped `src/lib/fsrs.ts` containing exactly
`// placeholder` while `Study.tsx` imported two functions from it — the app's
scheduling core, advertised in the README, never written.

The conjunction (hollow AND imported) is what makes this safe to block a turn
on, so most of these tests are about NOT firing: an empty package marker, a
sparse-but-real module, a stub nobody references.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode.agent.hollow_module import hollow_imported_modules, is_hollow_source


def _write(repo: Path, rel: str, text: str) -> str:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return rel


# ── the defect ──────────────────────────────────────────────────────────────


def test_the_anki_shape_is_caught(tmp_path: Path):
    _write(tmp_path, "src/lib/fsrs.ts", "// placeholder\n")
    _write(tmp_path, "src/pages/Study.tsx",
           "import { initCardState } from '../lib/fsrs';\nexport default function S(){}\n")
    assert hollow_imported_modules(
        str(tmp_path), ["src/lib/fsrs.ts", "src/pages/Study.tsx"]) == ["src/lib/fsrs.ts"]


def test_a_totally_empty_imported_module_is_caught(tmp_path: Path):
    _write(tmp_path, "src/lib/fsrs.ts", "")
    _write(tmp_path, "src/app.ts", "import './lib/fsrs';\n")
    assert hollow_imported_modules(str(tmp_path), ["src/lib/fsrs.ts"]) == ["src/lib/fsrs.ts"]


def test_block_comments_are_still_hollow(tmp_path: Path):
    _write(tmp_path, "src/lib/sched.ts", "/*\n * TODO: implement scheduling\n */\n")
    _write(tmp_path, "src/app.ts", "import { x } from './lib/sched';\n")
    assert hollow_imported_modules(str(tmp_path), ["src/lib/sched.ts"]) == ["src/lib/sched.ts"]


def test_python_stub_imported_by_package_path(tmp_path: Path):
    _write(tmp_path, "app/scheduler.py", "# placeholder\n")
    _write(tmp_path, "app/main.py", "from app.scheduler import review\n")
    assert hollow_imported_modules(str(tmp_path), ["app/scheduler.py"]) == ["app/scheduler.py"]


def test_absolute_changed_paths_work(tmp_path: Path):
    _write(tmp_path, "src/lib/fsrs.ts", "// placeholder\n")
    _write(tmp_path, "src/app.ts", "import './lib/fsrs';\n")
    got = hollow_imported_modules(str(tmp_path), [str(tmp_path / "src/lib/fsrs.ts")])
    assert got == ["src/lib/fsrs.ts"]


# ── everything that must NOT fire ───────────────────────────────────────────


def test_an_unreferenced_stub_is_not_flagged(tmp_path: Path):
    """Half the signal is not the signal. An empty file nobody imports is
    usually a scaffold the model has not wired up yet."""
    _write(tmp_path, "src/lib/fsrs.ts", "// placeholder\n")
    _write(tmp_path, "src/app.ts", "export const a = 1;\n")
    assert hollow_imported_modules(str(tmp_path), ["src/lib/fsrs.ts"]) == []


def test_empty_package_marker_is_never_flagged(tmp_path: Path):
    _write(tmp_path, "app/__init__.py", "")
    _write(tmp_path, "main.py", "from app import thing\n")
    assert hollow_imported_modules(str(tmp_path), ["app/__init__.py"]) == []


def test_a_sparse_but_real_module_is_not_flagged(tmp_path: Path):
    _write(tmp_path, "src/lib/fsrs.ts", "// scheduling\nexport const DAY = 86400;\n")
    _write(tmp_path, "src/app.ts", "import { DAY } from './lib/fsrs';\n")
    assert hollow_imported_modules(str(tmp_path), ["src/lib/fsrs.ts"]) == []


def test_a_docstring_only_python_module_is_not_flagged(tmp_path: Path):
    """A docstring is an intentional statement, not an unwritten module — and
    treating it as hollow would fire on plenty of legitimate code."""
    _write(tmp_path, "app/types.py", '"""Shared types."""\n')
    _write(tmp_path, "app/main.py", "from app.types import T\n")
    assert hollow_imported_modules(str(tmp_path), ["app/types.py"]) == []


def test_comment_markers_inside_strings_count_as_content(tmp_path: Path):
    _write(tmp_path, "src/u.ts", "export const p = '// not a comment';\n")
    _write(tmp_path, "src/app.ts", "import { p } from './u';\n")
    assert hollow_imported_modules(str(tmp_path), ["src/u.ts"]) == []


def test_non_code_files_are_ignored(tmp_path: Path):
    for rel in ("styles.css", "data.json", "notes.md"):
        _write(tmp_path, rel, "")
    assert hollow_imported_modules(str(tmp_path), ["styles.css", "data.json", "notes.md"]) == []


def test_declaration_files_are_ignored(tmp_path: Path):
    _write(tmp_path, "src/env.d.ts", "// generated\n")
    _write(tmp_path, "src/app.ts", "import './env';\n")
    assert hollow_imported_modules(str(tmp_path), ["src/env.d.ts"]) == []


def test_test_directories_are_ignored(tmp_path: Path):
    _write(tmp_path, "tests/helpers.py", "# placeholder\n")
    _write(tmp_path, "app/main.py", "from tests.helpers import h\n")
    assert hollow_imported_modules(str(tmp_path), ["tests/helpers.py"]) == []


def test_a_prose_mention_is_not_an_import(tmp_path: Path):
    """The importer scan must key on import syntax, not the bare word."""
    _write(tmp_path, "src/lib/fsrs.ts", "// placeholder\n")
    _write(tmp_path, "src/app.ts", "// we will add fsrs support later\nexport const a = 1;\n")
    assert hollow_imported_modules(str(tmp_path), ["src/lib/fsrs.ts"]) == []


def test_a_missing_file_is_not_a_defect(tmp_path: Path):
    assert hollow_imported_modules(str(tmp_path), ["src/gone.ts"]) == []


def test_paths_outside_the_repo_are_ignored(tmp_path: Path):
    outside = tmp_path.parent / "outside.ts"
    outside.write_text("// placeholder\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    assert hollow_imported_modules(str(repo), [str(outside)]) == []


def test_is_hollow_source_on_an_unreadable_path(tmp_path: Path):
    assert is_hollow_source(str(tmp_path / "nope.ts")) is False


# ── turn-level gate ─────────────────────────────────────────────────────────


def test_gate_blocks_and_bounds_its_nudges():
    from localcode.agent.hollow_module import HollowModuleGate
    gate = HollowModuleGate(max_retries=2)
    gate.mark(["src/lib/fsrs.ts"])
    assert gate.blocks_completion()
    assert gate.consume_retry() is True
    assert gate.consume_retry() is True
    assert gate.consume_retry() is False      # bounded — no infinite spin
    assert gate.blocks_completion()           # but the app is STILL hollow
    assert "src/lib/fsrs.ts" in gate.result_note()


def test_gate_clears_once_the_module_is_implemented():
    from localcode.agent.hollow_module import HollowModuleGate
    gate = HollowModuleGate()
    gate.mark(["src/lib/fsrs.ts"])
    gate.clear()
    assert not gate.blocks_completion() and gate.result_note() == ""


def test_loop_consults_the_hollow_gate_at_final_completion():
    """Wiring guard, mirroring the project-check gate's: the check is worthless
    if the completion decision never asks it."""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "localcode" / "agent" / "loop.py").read_text()
    assert "_hollow_gate.blocks_completion()" in src
    assert "_hollow_gate.result_note()" in src
