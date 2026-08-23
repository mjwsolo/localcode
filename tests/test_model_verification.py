"""The server must be asked WHICH MODEL it loaded — /health isn't enough.

Live evidence (2026-08-22): the user ran `/model` to switch to Muse Glimmer.
The status bar said "model: Muse Glimmer 30B UD-Q8_K_XL" and the model itself
played along, but the only llama-server on the machine was pid 49152 — an hour
old, serving `Qwen3.8-27B-UD-Q8_K_XL.gguf` on 8081. `ServerManager.start()`
never kills a foreign healthy server (by design — that would end another
terminal's session), so the healthcheck passed against somebody else's model
and every UI surface reported success.

These tests pin the fix: start() verifies via /props, fails loudly on a
mismatch, and the status bar never states an unverified model as fact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from localcode import server_manager as sm


# ── fake HTTP plumbing ─────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._body = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(routes: dict):
    """Serve `routes` (path-suffix → payload). Anything else raises, the way
    an older server without /props does."""
    def _open(url, timeout=None):
        for suffix, payload in routes.items():
            if url.endswith(suffix):
                return _FakeResponse(payload)
        raise OSError(f"no route for {url}")
    return _open


# ── (a) probe_loaded_model ─────────────────────────────────────────────

def test_probe_prefers_props_model_path(monkeypatch):
    monkeypatch.setattr(sm.urllib.request, "urlopen", _fake_urlopen({
        "/props": {"model_path": "/models/Qwen3.8-27B-UD-Q8_K_XL.gguf"},
        "/v1/models": {"data": [{"id": "something-else.gguf"}]},
    }))
    assert sm.probe_loaded_model(8081) == "/models/Qwen3.8-27B-UD-Q8_K_XL.gguf"


def test_probe_falls_back_to_v1_models(monkeypatch):
    monkeypatch.setattr(sm.urllib.request, "urlopen", _fake_urlopen({
        "/v1/models": {"data": [{"id": "Qwen3.8-27B-UD-Q8_K_XL.gguf"}]},
    }))
    assert sm.probe_loaded_model(8081) == "Qwen3.8-27B-UD-Q8_K_XL.gguf"


def test_probe_is_total_when_server_says_nothing(monkeypatch):
    monkeypatch.setattr(sm.urllib.request, "urlopen", _fake_urlopen({}))
    assert sm.probe_loaded_model(8081) is None


# ── (b) identity comparison ────────────────────────────────────────────

def test_identity_compares_basenames_case_sensitively():
    same = sm.model_identity_matches(
        "/a/b/Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
        "/completely/other/dir/Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
    )
    assert same
    assert not sm.model_identity_matches(
        "/a/Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
        "/a/Qwen3.8-27B-UD-Q8_K_XL.gguf",
    )
    assert not sm.model_identity_matches(
        "/a/Muse.gguf", "/a/muse.gguf",
    )
    assert not sm.model_identity_matches("", "/a/x.gguf")


# ── (c) start() verification ───────────────────────────────────────────

class _FakeProc:
    pid = 4242

    def poll(self):
        return None


@pytest.fixture()
def mgr(monkeypatch, tmp_path):
    """A ServerManager whose spawn/health/pidfile side effects are stubbed —
    only the verification step does real work."""
    m = sm.ServerManager.__new__(sm.ServerManager)
    import threading
    m._process = None
    m._model_path = None
    m._port = sm.DEFAULT_PORT
    m._pressure_thread = None
    m._lock = threading.Lock()
    m._last_exit_code = None
    m._last_death_was_pressure = False
    m._idle_timeout_s = 0.0
    m._last_activity_ts = 0.0
    m._idle_thread = None
    m._verified_model = None
    m._verification_error = None

    monkeypatch.setattr(sm.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(sm.ServerManager, "_shutdown_locked",
                        lambda self, reason="": None)
    monkeypatch.setattr(sm.ServerManager, "_pick_free_port",
                        lambda self, preferred: preferred)
    monkeypatch.setattr(sm.ServerManager, "_write_pid_file",
                        lambda self, pid: None)
    monkeypatch.setattr(sm.ServerManager, "_wait_healthy",
                        lambda self, port, timeout_s: True)
    monkeypatch.setattr(sm.ServerManager, "_ensure_idle_thread",
                        lambda self: None)
    return m


def test_start_fails_when_server_serves_a_different_model(mgr, monkeypatch):
    # The exact live shape: we ask for Muse Glimmer, port 8081 answers healthy,
    # but it's a prior session's Qwen server.
    monkeypatch.setattr(sm, "probe_loaded_model",
                        lambda port, timeout=2.0:
                        "/models/Qwen3.8-27B-UD-Q8_K_XL.gguf")
    ok = mgr.start(["llama-server", "--port", "8081"],
                   "/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf")
    assert ok is False
    err = mgr.verification_error
    assert err and "Muse-Glimmer-30B-UD-Q8_K_XL.gguf" in err
    assert "Qwen3.8-27B-UD-Q8_K_XL.gguf" in err
    # It reports what is ACTUALLY loaded, and never claims the requested one.
    assert Path(mgr.verified_model).name == "Qwen3.8-27B-UD-Q8_K_XL.gguf"


def test_start_succeeds_when_the_right_model_is_loaded(mgr, monkeypatch):
    monkeypatch.setattr(sm, "probe_loaded_model",
                        lambda port, timeout=2.0:
                        "/other/dir/Muse-Glimmer-30B-UD-Q8_K_XL.gguf")
    ok = mgr.start(["llama-server", "--port", "8081"],
                   "/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf")
    assert ok is True
    assert mgr.verification_error is None
    assert Path(mgr.verified_model).name == "Muse-Glimmer-30B-UD-Q8_K_XL.gguf"


def test_start_survives_a_server_that_cannot_answer(mgr, monkeypatch):
    # Unknown is not a mismatch: an older server that has no /props must not
    # brick the launch — it just stays unverified.
    monkeypatch.setattr(sm, "probe_loaded_model", lambda port, timeout=2.0: None)
    ok = mgr.start(["llama-server", "--port", "8081"], "/models/Muse.gguf")
    assert ok is True
    assert mgr.verified_model is None
    assert mgr.verification_error is None


def test_restart_goes_through_the_same_verification(mgr, monkeypatch):
    monkeypatch.setattr(sm, "probe_loaded_model",
                        lambda port, timeout=2.0: "/m/Qwen3.8-27B-UD-Q8_K_XL.gguf")
    assert mgr.restart(["llama-server", "--port", "8081"], "/m/Muse.gguf") is False
    assert mgr.verification_error is not None


# ── (d) the status bar must not claim an unverified model ──────────────

class _FakeRuntime:
    provider = "llama_cpp"
    model = "/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf"
    internal_thinking_mode = "off"


class _FakeConfig:
    runtime = _FakeRuntime()


class _FakeTui:
    config = _FakeConfig()


class _FakeScreen:
    """Minimal stand-in for ChatScreen — we exercise the two real methods
    (`_verify_serving_model`, `_update_status`) against it."""
    tui = _FakeTui()

    def __init__(self):
        self._context_used = 0
        self._context_max = 100000
        self._server_restarting = False
        self._server_alive = True
        self._verified_model_name = None
        self._model_mismatch = None
        self.rendered = ""
        self.app = type("A", (), {"size": type("S", (), {"width": 200})()})()

    # widget plumbing
    def query_one(self, sel, kind=None):
        screen = self

        class _Bar:
            size = type("S", (), {"width": 200})()

            def update(self, text):
                screen.rendered = text.plain if hasattr(text, "plain") else str(text)
        return _Bar()

    def _model_supports_thinking(self, cur):
        return True

    def _permissions_label(self):
        return "on"


def _screen_methods():
    from localcode.tui.screens.chat import ChatScreen
    return ChatScreen._verify_serving_model, ChatScreen._update_status


def test_status_bar_shouts_when_a_different_model_is_serving(monkeypatch):
    verify, update = _screen_methods()
    import localcode.server_manager as _sm
    monkeypatch.setattr(_sm, "probe_loaded_model",
                        lambda port, timeout=1.0: "/m/Qwen3.8-27B-UD-Q8_K_XL.gguf")
    monkeypatch.setattr(_sm.ServerManager, "get",
                        classmethod(lambda cls: _FakeStubMgr()))
    s = _FakeScreen()
    verify(s)
    assert s._model_mismatch is not None
    assert s._verified_model_name is None
    update(s)
    # Loud on the left (survives narrow-terminal truncation) AND names the
    # model that is really answering.
    assert "server: WRONG MODEL" in s.rendered
    assert "serving Qwen3.8-27B-UD-Q8_K_XL" in s.rendered


def test_status_bar_names_the_model_once_verified(monkeypatch):
    verify, update = _screen_methods()
    import localcode.server_manager as _sm
    monkeypatch.setattr(
        _sm, "probe_loaded_model",
        lambda port, timeout=1.0: "/m/Muse-Glimmer-30B-UD-Q8_K_XL.gguf")
    monkeypatch.setattr(_sm.ServerManager, "get",
                        classmethod(lambda cls: _FakeStubMgr()))
    s = _FakeScreen()
    verify(s)
    assert s._model_mismatch is None
    assert s._verified_model_name == "Muse-Glimmer-30B-UD-Q8_K_XL.gguf"
    update(s)
    assert "WRONG MODEL" not in s.rendered
    assert "unverified" not in s.rendered
    assert "server: ready" in s.rendered


def test_status_bar_says_unverified_when_the_probe_is_silent(monkeypatch):
    verify, update = _screen_methods()
    import localcode.server_manager as _sm
    monkeypatch.setattr(_sm, "probe_loaded_model", lambda port, timeout=1.0: None)
    monkeypatch.setattr(_sm.ServerManager, "get",
                        classmethod(lambda cls: _FakeStubMgr()))
    s = _FakeScreen()
    verify(s)
    update(s)
    assert "unverified" in s.rendered
    assert "WRONG MODEL" not in s.rendered


class _FakeStubMgr:
    """ServerManager stand-in: knows what was requested and which port."""
    current_model = "/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf"
    port = 8081
