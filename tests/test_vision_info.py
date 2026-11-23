"""vision_info() drives the TUI's image-without-vision dialog."""
from types import SimpleNamespace

from localcode.ui import server_cmd
from localcode.ui.supervisor import Supervisor


def _sup(current):
    s = Supervisor.__new__(Supervisor)
    s.current = current
    s._model_path = lambda alias: f"/models/{alias}.gguf"
    return s


def test_no_model_loaded():
    assert _sup(None).vision_info() == {"vision": False, "vision_available": False, "vision_size_gb": 0.0}


def test_vision_missing_but_downloadable(monkeypatch):
    choice = SimpleNamespace(supports_vision=True, mmproj_size_gb=0.9)
    monkeypatch.setattr(server_cmd, "catalog_choice", lambda p: choice)
    monkeypatch.setattr(server_cmd, "mmproj_for", lambda p: None)
    assert _sup("m").vision_info() == {"vision": False, "vision_available": True, "vision_size_gb": 0.9}


def test_vision_unavailable_for_model(monkeypatch):
    choice = SimpleNamespace(supports_vision=False, mmproj_size_gb=0.0)
    monkeypatch.setattr(server_cmd, "catalog_choice", lambda p: choice)
    monkeypatch.setattr(server_cmd, "mmproj_for", lambda p: None)
    assert _sup("m").vision_info() == {"vision": False, "vision_available": False, "vision_size_gb": 0.0}


def test_vision_ready(monkeypatch):
    choice = SimpleNamespace(supports_vision=True, mmproj_size_gb=0.9)
    monkeypatch.setattr(server_cmd, "catalog_choice", lambda p: choice)
    monkeypatch.setattr(server_cmd, "mmproj_for", lambda p: "/models/mmproj.gguf")
    assert _sup("m").vision_info()["vision"] is True
