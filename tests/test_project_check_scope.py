"""Project checks are scoped to the package the user launched in, never the
whole git root.

Reproduces 2026-09-08: localcode launched in a subdirectory of a larger clone
ran ruff over the entire clone, the post-edit nudge quoted findings from an
unrelated tree, and the model went and edited those files.
"""
from __future__ import annotations

from pathlib import Path

from localcode.tools.project_check import (
    _confine_lines,
    check_root_for,
    run_project_check_result,
)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


def test_check_root_is_launch_dir_when_no_package_marker(tmp_path):
    repo = _repo(tmp_path)
    ws = repo / "bench" / "sandbox" / "workspace"
    ws.mkdir(parents=True)
    assert Path(check_root_for(ws, repo)) == ws


def test_check_root_climbs_to_nearest_package_but_not_past_repo(tmp_path):
    repo = _repo(tmp_path)
    pkg = repo / "packages" / "api"
    (pkg / "src").mkdir(parents=True)
    (pkg / "pyproject.toml").write_text("")
    assert Path(check_root_for(pkg / "src", repo)) == pkg
    assert Path(check_root_for(repo, repo)) == repo


def test_check_root_at_repo_root_is_unchanged_behaviour(tmp_path):
    repo = _repo(tmp_path)
    (repo / "pyproject.toml").write_text("")
    assert Path(check_root_for(repo, repo)) == repo


def test_confine_drops_absolute_paths_outside_root():
    lines = [
        "[ruff] reported errors:",
        "in/app.py:3:1: F821 Undefined name `x`",  # relative → inside
        "/repo/tasks/047/fixtures/api.py:5:12: F821 Undefined name `y`",  # elsewhere
        "/repo/ws/in/b.py:1:1: E999 SyntaxError",  # inside by prefix
    ]
    kept = _confine_lines(lines, "/repo/ws")
    assert "/repo/tasks/047/fixtures/api.py:5:12: F821 Undefined name `y`" not in kept
    assert len(kept) == 3


def test_checker_scoped_to_launch_dir_ignores_sibling_trees(tmp_path):
    """End to end: a broken file in a sibling tree must not surface when the
    check is run from the launch dir's check root."""
    repo = _repo(tmp_path)
    other = repo / "tasks" / "047" / "fixtures"
    other.mkdir(parents=True)
    (other / "api.py").write_text("def f(:\n")  # syntax error: caught by ruff AND compileall
    ws = repo / "bench" / "workspace"
    ws.mkdir(parents=True)
    (ws / "ok.py").write_text("x = 1\n")

    scoped = run_project_check_result(check_root_for(ws, repo))
    assert not scoped.is_red, scoped.detail

    whole = run_project_check_result(str(repo))
    assert whole.is_red  # the old behaviour: the sibling's error leaks in
    assert "api.py" in whole.detail
