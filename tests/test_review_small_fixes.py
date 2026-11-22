"""Small reliability fixes: bind errors, discovery limits, empty weights."""
import errno
import threading

import pytest

from localcode import model_files
from localcode.ui import ports
from localcode.ui.supervisor import Supervisor


def test_denied_bind_keeps_the_original_error(monkeypatch):
    class Denied:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def bind(self, addr): raise PermissionError(errno.EPERM, "Operation not permitted")
    monkeypatch.setattr(ports, "preferred_port", lambda: None)
    monkeypatch.setattr(ports.socket, "socket", lambda *a, **k: Denied())
    monkeypatch.setattr(ports, "_status", lambda port: (_ for _ in ()).throw(OSError("no status")))
    with pytest.raises(RuntimeError) as e:
        ports.choose_ports()
    assert "not permitted" in str(e.value) and "No free" not in str(e.value)


def test_each_search_root_gets_its_own_directory_budget(tmp_path, monkeypatch):
    big, small = tmp_path / "big", tmp_path / "small"
    big.mkdir(); small.mkdir()
    for i in range(30):
        (big / f"d{i}").mkdir()
    (small / "late.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(model_files, "search_roots", lambda _: [big, small])
    # Push the big root past the per-root directory budget.
    for i in range(5001 - 31):
        (big / f"d{i % 30}" / f"e{i}").mkdir()
    assert "late.gguf" in model_files.model_files(big)


def test_zero_byte_gguf_is_not_ready(tmp_path):
    sup = Supervisor.__new__(Supervisor)
    sup.models_dir = tmp_path
    (tmp_path / "empty.gguf").touch()
    (tmp_path / "real.gguf").write_bytes(b"GGUF")
    assert sup._model_ready("empty.gguf") is False
    assert sup._model_ready("real.gguf") is True
