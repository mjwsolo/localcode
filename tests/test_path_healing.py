"""Regression: heal model-corrupted absolute paths back onto repo_root.

Observed live (2026-07-13): the model turned its workdir
`…/localcode-evals/20260713-024502/tempconv` into
`…/localcode-evals/2026-0713-0245-02/tempconv` (hyphens hallucinated into the
timestamp) and wrote the entire project into the phantom tree while the real
workdir stayed empty. ToolContext.resolve_path must remap near-miss prefixes
onto the real repo_root, and must NOT touch genuinely different locations.
"""
import types
from pathlib import Path

from localcode.tools.base import ToolContext


def _ctx(root: Path) -> ToolContext:
    app = types.SimpleNamespace(repo_root=root)
    return ToolContext(app=app, out=None)  # type: ignore[arg-type]


def test_heals_hyphen_corrupted_timestamp_dir(tmp_path: Path) -> None:
    root = tmp_path / "localcode-evals" / "20260713-024502" / "tempconv"
    root.mkdir(parents=True)
    ctx = _ctx(root)
    corrupted = str(tmp_path / "localcode-evals" / "2026-0713-0245-02" / "tempconv" / "pyproject.toml")
    healed = ctx.resolve_path(corrupted)
    assert healed == root / "pyproject.toml"


def test_heals_deeper_corrupted_paths(tmp_path: Path) -> None:
    root = tmp_path / "projects" / "my-anki-app"
    root.mkdir(parents=True)
    ctx = _ctx(root)
    corrupted = str(tmp_path / "projects" / "my-anki-ap" / "src" / "db.js")
    assert ctx.resolve_path(corrupted) == root / "src" / "db.js"


def test_relative_and_in_root_paths_untouched(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    ctx = _ctx(root)
    assert ctx.resolve_path("src/main.py") == root / "src" / "main.py"
    assert ctx.resolve_path(str(root / "src" / "main.py")) == root / "src" / "main.py"


def test_genuinely_different_location_respected(tmp_path: Path) -> None:
    # LocalCode has full filesystem access by design: a path that is NOT a
    # near-miss of repo_root must pass through unchanged.
    root = tmp_path / "proj"
    root.mkdir(parents=True)
    ctx = _ctx(root)
    other = str(tmp_path / "completely" / "elsewhere" / "notes.txt")
    assert str(ctx.resolve_path(other)) == other
