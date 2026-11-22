"""Choosing a models folder: never touch an existing file, never claim a setting
that was not saved."""
import threading
from pathlib import Path

import pytest

from localcode import config
from localcode.ui.supervisor import Supervisor


@pytest.fixture
def sup(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_MODEL_DIR", str(tmp_path / "old"))
    s = Supervisor.__new__(Supervisor)
    s.lock = threading.Lock()
    s.active = {}
    s.state = {"state": "idle"}
    s.models_dir = tmp_path / "old"
    return s


def test_existing_probe_named_file_is_left_alone(sup, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "save_config", lambda cfg: None)
    target = tmp_path / "models"
    target.mkdir()
    keep = target / ".localcode-write-test"
    keep.write_text("mine")
    assert sup.set_models_dir(str(target)).get("ok") is True
    assert keep.read_text() == "mine"
    assert sorted(p.name for p in target.iterdir()) == [".localcode-write-test"]


def test_unsaved_setting_is_an_error_and_changes_nothing(sup, tmp_path, monkeypatch):
    def boom(cfg):
        raise OSError("disk full")
    monkeypatch.setattr(config, "save_config", boom)
    result = sup.set_models_dir(str(tmp_path / "new"))
    assert "error" in result and "disk full" in result["error"]
    assert sup.models_dir == tmp_path / "old"
    assert Path(__import__("os").environ["LOCALCODE_MODEL_DIR"]) == tmp_path / "old"
