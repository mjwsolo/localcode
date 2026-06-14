"""Steering tests: the agent must write code in the project's actual
language/conventions and not fall back to Python idioms.

Covers (a) the static SYSTEM_PROMPT language-matching rule and (b) the
dynamic per-repo "Project stack:" line emitted from repo-root marker
files (package.json/tsconfig.json/go.mod/Cargo.toml/pyproject.toml/…).
"""
from __future__ import annotations

from pathlib import Path

from localcode.agent.prompts import SYSTEM_PROMPT, project_stack_line


def test_system_prompt_has_language_matching_rule() -> None:
    text = SYSTEM_PROMPT.lower()
    # The rule must forbid Python syntax leaking into other languages and
    # tell the model to match the file's actual language + conventions.
    assert "match the project's language" in text
    assert "triple-quoted docstrings" in text
    assert "snake_case" in SYSTEM_PROMPT
    assert ".tsx" in SYSTEM_PROMPT
    assert "conventions" in text


def test_project_stack_line_detects_typescript_react(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name":"app"}\n')
    (tmp_path / "tsconfig.json").write_text("{}\n")

    line = project_stack_line(tmp_path)

    assert line.startswith("Project stack: ")
    assert "package.json" in line
    assert "tsconfig.json" in line
    assert "follow its conventions." in line
    assert line.endswith("\n")
    # One line only — must not bloat / break the cached prefix.
    assert line.count("\n") == 1


def test_project_stack_line_detects_go(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/x\n")
    line = project_stack_line(tmp_path)
    assert line.startswith("Project stack: Go (go.mod present)")


def test_project_stack_line_empty_when_no_markers(tmp_path: Path) -> None:
    # No marker files -> empty string keeps the prompt prefix byte-identical.
    assert project_stack_line(tmp_path) == ""


def test_project_stack_line_never_raises_on_bad_path() -> None:
    # Best-effort: a non-existent / unreadable root yields "" not an error.
    assert project_stack_line(Path("/nonexistent/repo/root/xyz")) == ""
