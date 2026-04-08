"""Tests for gem.context — repo root detection, file reading, git helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gem.context import (
    IGNORE_DIRS,
    build_context_block,
    find_repo_root,
    git_diff,
    git_status,
    list_repo_files,
    read_file,
)


class TestFindRepoRoot:
    """Verify find_repo_root walks up to the nearest .git directory."""

    def test_finds_git_root(self, tmp_repo: Path) -> None:
        found = find_repo_root(tmp_repo / "sub")
        assert found == tmp_repo

    def test_finds_from_root_itself(self, tmp_repo: Path) -> None:
        found = find_repo_root(tmp_repo)
        assert found == tmp_repo

    def test_returns_start_when_no_git(self, tmp_path: Path) -> None:
        """When no .git exists anywhere, return the start directory resolved."""
        no_git = tmp_path / "nope"
        no_git.mkdir()
        found = find_repo_root(no_git)
        assert found == no_git.resolve()


class TestListRepoFiles:
    """Verify list_repo_files returns files and respects ignore dirs."""

    def test_lists_committed_files(self, tmp_repo: Path) -> None:
        files = list_repo_files(tmp_repo)
        assert "main.py" in files
        assert "utils.py" in files
        # Submodule file
        found_sub = [f for f in files if "module.py" in f]
        assert len(found_sub) == 1

    def test_ignores_pycache(self, tmp_repo: Path) -> None:
        pycache = tmp_repo / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-312.pyc").write_bytes(b"")
        files = list_repo_files(tmp_repo)
        assert not any("__pycache__" in f for f in files)

    def test_pattern_filter(self, tmp_repo: Path) -> None:
        files = list_repo_files(tmp_repo, pattern="main")
        assert "main.py" in files
        assert "utils.py" not in files

    def test_limit_respected(self, tmp_repo: Path) -> None:
        files = list_repo_files(tmp_repo, limit=1)
        assert len(files) <= 1

    def test_ignore_dirs_constant(self) -> None:
        """Verify the expected directories are in the ignore set."""
        assert ".git" in IGNORE_DIRS
        assert "node_modules" in IGNORE_DIRS
        assert "__pycache__" in IGNORE_DIRS


class TestReadFile:
    """Verify read_file returns content and handles truncation."""

    def test_reads_existing_file(self, tmp_repo: Path) -> None:
        content = read_file(tmp_repo, "main.py")
        assert "def hello" in content

    def test_truncates_large_file(self, tmp_repo: Path) -> None:
        big = "x" * 20000
        (tmp_repo / "big.txt").write_text(big)
        content = read_file(tmp_repo, "big.txt", max_chars=100)
        assert len(content) < 200
        assert "truncated" in content.lower()

    def test_raises_on_missing_file(self, tmp_repo: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_file(tmp_repo, "nonexistent.py")


class TestGitStatus:
    """Verify git_status reports clean/dirty state."""

    def test_clean_repo(self, tmp_repo: Path) -> None:
        status = git_status(tmp_repo)
        assert status == "clean"

    def test_dirty_repo(self, tmp_repo: Path) -> None:
        (tmp_repo / "new_file.txt").write_text("new content")
        status = git_status(tmp_repo)
        assert "new_file.txt" in status
        assert "?" in status  # untracked marker


class TestGitDiff:
    """Verify git_diff returns diff output for modified files."""

    def test_no_diff_when_clean(self, tmp_repo: Path) -> None:
        diff = git_diff(tmp_repo)
        assert "No working tree diff" in diff

    def test_diff_after_modification(self, tmp_repo: Path) -> None:
        (tmp_repo / "main.py").write_text("def hello():\n    return 'updated'\n")
        diff = git_diff(tmp_repo)
        assert "updated" in diff
        assert "diff" in diff.lower() or "---" in diff

    def test_diff_truncation(self, tmp_repo: Path) -> None:
        (tmp_repo / "main.py").write_text("x" * 50000)
        diff = git_diff(tmp_repo, max_chars=100)
        assert len(diff) < 200


class TestBuildContextBlock:
    """Verify build_context_block assembles a context string."""

    def test_includes_repo_root(self, tmp_repo: Path) -> None:
        block = build_context_block(tmp_repo, [], max_chars=10000)
        assert str(tmp_repo) in block

    def test_includes_git_status(self, tmp_repo: Path) -> None:
        block = build_context_block(tmp_repo, [], max_chars=10000)
        assert "Git status" in block

    def test_includes_pinned_files(self, tmp_repo: Path) -> None:
        block = build_context_block(tmp_repo, ["main.py"], max_chars=10000)
        assert "def hello" in block
        assert "main.py" in block

    def test_handles_missing_pinned_file(self, tmp_repo: Path) -> None:
        block = build_context_block(tmp_repo, ["missing.py"], max_chars=10000)
        assert "File missing" in block

    def test_truncation(self, tmp_repo: Path) -> None:
        block = build_context_block(tmp_repo, ["main.py"], max_chars=50)
        assert len(block) <= 70  # some slack for truncation message
