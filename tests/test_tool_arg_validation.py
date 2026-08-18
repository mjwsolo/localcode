"""Edge-case hardening for the file tools: a small model that drops a required
arg, or points a text tool at a binary/directory, must get a clear recoverable
message — never an unhandled crash or a silent destructive write.
"""
from pathlib import Path

from localcode.tools import read_file, write_file, edit_file
from localcode.tools.base import ToolContext


class _App:
    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.session = type("_S", (), {"current_task": None})()


class _Out:
    def __getattr__(self, _n):
        return lambda *a, **k: None


def _ctx(tmp_path):
    return ToolContext(app=_App(tmp_path), out=_Out())


def test_edit_missing_new_string_is_clear_error_not_crash(tmp_path):
    (tmp_path / "x.py").write_text("hello world\n")
    out = edit_file.execute(_ctx(tmp_path), {"path": "x.py", "old_string": "hello"})
    assert "new_string" in out and out.startswith("Error")
    # file untouched
    assert (tmp_path / "x.py").read_text() == "hello world\n"


def test_edit_missing_old_string_is_clear_error(tmp_path):
    (tmp_path / "x.py").write_text("hello\n")
    out = edit_file.execute(_ctx(tmp_path), {"path": "x.py", "new_string": "hi"})
    assert "old_string" in out and out.startswith("Error")


def test_write_missing_content_key_refuses(tmp_path):
    # A missing content key must NOT silently create/overwrite with "".
    (tmp_path / "keep.py").write_text("important\n")
    out = write_file.execute(_ctx(tmp_path), {"path": "keep.py"})
    assert "content" in out and out.startswith("Error")
    assert (tmp_path / "keep.py").read_text() == "important\n"  # not clobbered


def test_write_explicit_empty_content_is_allowed(tmp_path):
    out = write_file.execute(_ctx(tmp_path), {"path": "empty.py", "content": ""})
    assert not out.startswith("Error")
    assert (tmp_path / "empty.py").exists()


def test_read_binary_file_is_refused_not_dumped(tmp_path):
    (tmp_path / "b.dat").write_bytes(bytes(range(256)))
    out = read_file.execute(_ctx(tmp_path), {"path": "b.dat"})
    assert "BINARY" in out.upper()
    # the raw high bytes are NOT dumped into the result
    assert "\x00" not in out


def test_read_text_file_still_works(tmp_path):
    (tmp_path / "t.py").write_text("print('ok')\n")
    out = read_file.execute(_ctx(tmp_path), {"path": "t.py"})
    assert "print('ok')" in out
