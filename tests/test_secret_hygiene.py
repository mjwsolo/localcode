"""Secret-hygiene regressions.

Three real leaks, three guards:
  1. `<repo>/.localcode/` holds full prompt/response transcripts inside the
     user's own git repo and was never self-ignored.
  2. State files under `~/.localcode` were world-readable (0644 in a 0755 dir).
  3. The lexical index copied `.env` / key material verbatim to
     `~/.localcode/indexes/<sha1>.json`.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from localcode import paths


# ── 1. project state dir self-ignores ────────────────────────────────

def test_project_state_dir_writes_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    d = paths.project_state_dir(tmp_path)
    assert d == tmp_path / ".localcode"
    gitignore = d / ".gitignore"
    assert gitignore.exists(), "a fresh .localcode/ must ignore itself"
    assert "*" in gitignore.read_text().split()


def test_project_state_dir_repairs_existing_dir(tmp_path: Path) -> None:
    """An upgrade from an older version has the dir but no .gitignore."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".localcode").mkdir()
    paths.project_state_dir(tmp_path)
    assert (tmp_path / ".localcode" / ".gitignore").exists()


def test_project_state_dir_does_not_clobber_user_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    d = tmp_path / ".localcode"
    d.mkdir()
    (d / ".gitignore").write_text("!keepme\n")
    paths.project_state_dir(tmp_path)
    assert (d / ".gitignore").read_text() == "!keepme\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_project_state_dir_is_owner_only(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    d = paths.project_state_dir(tmp_path)
    assert stat.S_IMODE(d.stat().st_mode) == 0o700


# ── 2. private writes ────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_write_private_is_owner_only(tmp_path: Path) -> None:
    p = tmp_path / "secret.txt"
    paths.write_private(p, "api_key = 'sk-live'")
    assert p.read_text() == "api_key = 'sk-live'"
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_write_private_tightens_preexisting_world_readable_file(tmp_path: Path) -> None:
    p = tmp_path / "secret.txt"
    p.write_text("old")
    os.chmod(p, 0o644)
    paths.write_private(p, "new")
    assert p.read_text() == "new"
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_save_config_writes_owner_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path / "home"))
    from localcode import config as config_mod

    cfg = config_mod.load_config()
    path = config_mod.save_config(cfg)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    home = config_mod.get_home_dir()
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert stat.S_IMODE((home / "sessions").stat().st_mode) == 0o700


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_history_db_is_owner_only(tmp_path: Path) -> None:
    from localcode.history import HistoryDB

    db_path = tmp_path / "history.db"
    HistoryDB(str(db_path))
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600


def test_chmod_quiet_swallows_missing_file(tmp_path: Path) -> None:
    paths.chmod_quiet(tmp_path / "nope", 0o600)  # must not raise


# ── 3. the index skips credentials ───────────────────────────────────

@pytest.mark.parametrize("name", [
    ".env", ".env.local", ".env.production",
    "server.pem", "app.key", "store.p12", "release.keystore",
    "id_rsa", ".netrc", "credentials.json",
])
def test_is_secret_file_flags_credentials(name: str) -> None:
    from localcode.indexer import _is_secret_file

    assert _is_secret_file(name)
    assert _is_secret_file(f"config/{name}")


@pytest.mark.parametrize("name", [
    "tokenizer.py", "api_keys.py", "environment.py", "env.py",
    "keymap.ts", "main.go", "README.md", "Dockerfile",
])
def test_is_secret_file_allows_ordinary_source(name: str) -> None:
    from localcode.indexer import _is_secret_file

    assert not _is_secret_file(name)
    assert not _is_secret_file(f"src/{name}")


def test_build_index_skips_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path / "home"))
    from localcode import indexer

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("OPENAI_API_KEY=sk-live-DO-NOT-INDEX\n")
    (repo / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nDO-NOT-INDEX\n")
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")

    _, index_file = indexer.build_index(repo)
    blob = index_file.read_text()
    assert "DO-NOT-INDEX" not in blob
    assert "sk-live" not in blob
    assert "hello" in blob, "ordinary source must still be indexed"
    if sys.platform != "win32":
        assert stat.S_IMODE(index_file.stat().st_mode) == 0o600
