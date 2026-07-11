"""Tests for localcode.context — repo root detection, file reading, git helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from localcode.context import (
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

    def test_prefers_project_marker_over_walking_to_home(self, tmp_path: Path) -> None:
        """No .git, but a pyproject.toml in an ancestor pins that project dir —
        instead of returning the (deep) launch dir or walking up to $HOME."""
        project = tmp_path / "myproj"
        (project / "src" / "pkg").mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\nname='x'\n")
        # Launched deep inside the project, no git anywhere.
        found = find_repo_root(project / "src" / "pkg")
        assert found == project.resolve()

    def test_nearest_marker_wins(self, tmp_path: Path) -> None:
        """When markers exist at multiple levels, the NEAREST (deepest) wins."""
        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        (outer / "package.json").write_text("{}")
        (inner / "package.json").write_text("{}")
        assert find_repo_root(inner) == inner.resolve()

    def test_git_root_beats_nearer_marker(self, tmp_path: Path) -> None:
        """A real .git root always wins, even when a shallower marker exists —
        behaviour must stay IDENTICAL when a git checkout is present."""
        repo = tmp_path / "repo"
        deep = repo / "frontend"
        deep.mkdir(parents=True)
        (repo / ".git").mkdir()
        # A package.json sits in the deeper dir, but .git at the repo root wins.
        (deep / "package.json").write_text("{}")
        assert find_repo_root(deep) == repo.resolve()

    def test_never_adopts_home_via_marker(self, tmp_path: Path, monkeypatch) -> None:
        """A marker sitting at $HOME must NOT be adopted as the repo root —
        $HOME is never a legitimate project root."""
        fake_home = tmp_path / "home"
        launch = fake_home / "Desktop" / "build"
        launch.mkdir(parents=True)
        (fake_home / "package.json").write_text("{}")  # marker AT home
        monkeypatch.setenv("HOME", str(fake_home))
        # Walk stops at $HOME, so the home-level marker is refused; falls back
        # to the launch dir rather than adopting $HOME.
        found = find_repo_root(launch)
        assert found == launch.resolve()
        assert found != fake_home.resolve()

    def test_is_home_or_shallower_flags_home(self, tmp_path: Path, monkeypatch) -> None:
        from localcode.context import _is_home_or_shallower
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        assert _is_home_or_shallower(fake_home.resolve()) is True
        assert _is_home_or_shallower(Path(fake_home.anchor)) is True
        project = fake_home / "Github" / "proj"
        project.mkdir(parents=True)
        assert _is_home_or_shallower(project.resolve()) is False


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


def test_window_aware_compaction_scales_with_ram():
    # Compaction must be DYNAMIC per machine: tiny window -> aggressive (the
    # legacy 36KB/keep-4); big window -> keep far more history (a 128GB Mac was
    # crushing its 128K window to 36KB, making the model lose track). None ->
    # legacy fixed behaviour.
    from localcode.agent.context import _window_aware_compaction as W
    assert W(None) == (36_000, 4)
    small_b, small_k = W(int(16384 * 3.5))      # 16K window
    big_b, big_k = W(int(131072 * 3.5))         # 128K window
    assert small_b == 36_000 and small_k == 4   # small stays aggressive
    assert big_b > 200_000 and big_k >= 16      # big keeps much more
    assert big_b > small_b and big_k > small_k  # strictly scales up


def test_progress_ledger_dedups_and_shows_outcomes():
    from localcode.agent.context import build_progress_ledger
    led = build_progress_ledger(
        changed_files=["src/types.ts"],
        bash_history=[("npm i", "added 5 pkgs"), ("npm run build", "[exit code 1] TS2304")],
        files_read=["a.ts", "b.ts", "a.ts", "a.ts"],  # a.ts re-read
        budget_chars=3500,
    )
    # Header frames the ledger as the assistant's OWN actions (not the user's
    # work / pre-existing files) so the model doesn't treat stale dirs as its task.
    assert "your own tool calls" in led.lower() and "new progress" in led.lower()
    assert led.count("a.ts") == 1  # deduped
    assert "src/types.ts" in led
    assert "ok npm i" in led and "x npm run build" in led  # outcomes flagged
    # Empty state -> empty ledger (stable first-round prefix).
    assert build_progress_ledger([], [], [], 3500) == ""


def test_progress_ledger_budget_is_ram_tiered():
    from localcode.model_config import progress_ledger_budget_chars
    assert progress_ledger_budget_chars(65536) == 1200    # 16 GB / 64K: tight
    assert progress_ledger_budget_chars(98304) == 1600    # 48 GB / 96K
    assert progress_ledger_budget_chars(131072) == 2200   # 64 GB / 128K
    assert progress_ledger_budget_chars(262144) == 3500   # 128 GB / 256K: rich
    # Budget trims the OPTIONAL content (files-read / commands), but the
    # created/edited list is PROTECTED — never dropped, since it's the anchor
    # that stops the model forgetting what it built. So a long read/command
    # payload gets trimmed while the created file stays.
    from localcode.agent.context import build_progress_ledger
    big = build_progress_ledger(["kept.py"], [("c" * 300, "ok")], ["p" * 300], 100)
    assert "kept.py" in big                       # created file never dropped
    assert "cccc" not in big or "…" in big        # long command payload trimmed
