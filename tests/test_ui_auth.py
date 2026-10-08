"""The two local listeners refuse anything without the session secrets.

A web page in the user's browser can send requests to localhost ports, so
loopback binding alone is not a boundary. These tests drive the real control
handler over a real socket.
"""
from __future__ import annotations

import json
import os
import stat
import threading
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from localcode.ui import auth
from localcode.ui import supervisor as sv

TOKEN = "t" * 43


@pytest.fixture()
def control():
    """A live control server backed by a stub supervisor."""
    calls: list[str] = []
    sup = types.SimpleNamespace(
        control_token=TOKEN,
        control_port=0,
        models_dir_info=lambda: {"path": "/models"},
        cancel_warmup=lambda: (calls.append("warmup/cancel"), {"ok": True})[1],
        voice_stop=lambda: (calls.append("voice/stop"), {"text": "secret words"})[1],
        decision=lambda body: (calls.append("decision"), {"answers": {"answer": {"noul": 0.9}}})[1],
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), sv.make_handler(sup))
    sup.control_port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield sup.control_port, calls
    finally:
        httpd.shutdown()
        httpd.server_close()


def _call(port, path, *, method="GET", headers=None, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def test_read_without_token_is_refused(control):
    port, _ = control
    code, body = _call(port, "/models_dir")
    assert code == 403 and "token" in body["error"]


def test_read_with_token_succeeds(control):
    port, _ = control
    code, body = _call(port, "/models_dir", headers=auth.control_headers(TOKEN))
    assert code == 200 and body == {"path": "/models"}


def test_wrong_token_is_refused(control):
    port, _ = control
    code, _ = _call(port, "/models_dir", headers=auth.control_headers("x" * 43))
    assert code == 403


def test_browser_request_is_refused_even_with_the_token(control):
    """A page cannot know the token, but if it ever did, Origin still refuses it."""
    port, calls = control
    headers = {**auth.control_headers(TOKEN), "Origin": "https://evil.example", "Content-Type": "application/json"}
    code, body = _call(port, "/voice/stop", method="POST", headers=headers, body=b"{}")
    assert code == 403 and "cross-origin" in body["error"] and calls == []


def test_rebound_host_is_refused(control):
    """DNS rebinding: the request arrives on loopback but names another host."""
    port, calls = control
    headers = {**auth.control_headers(TOKEN), "Host": f"evil.example:{port}", "Content-Type": "application/json"}
    code, body = _call(port, "/voice/stop", method="POST", headers=headers, body=b"{}")
    assert code == 403 and "Host" in body["error"] and calls == []


def test_csrf_style_post_is_refused(control):
    """A no-cors fetch or a form can only send text/plain or form encodings."""
    port, calls = control
    headers = {**auth.control_headers(TOKEN), "Content-Type": "text/plain"}
    code, body = _call(port, "/voice/stop", method="POST", headers=headers, body=b"{}")
    assert code == 403 and "application/json" in body["error"] and calls == []


def test_post_without_token_never_reaches_the_supervisor(control):
    port, calls = control
    code, _ = _call(port, "/voice/stop", method="POST", headers={"Content-Type": "application/json"}, body=b"{}")
    assert code == 403 and calls == []


def test_authorised_post_runs(control):
    port, calls = control
    headers = {**auth.control_headers(TOKEN), "Content-Type": "application/json"}
    code, body = _call(port, "/warmup/cancel", method="POST", headers=headers, body=b"{}")
    assert code == 200 and body == {"ok": True} and calls == ["warmup/cancel"]


@pytest.mark.parametrize("extra", [{}, {"Origin": "https://evil.example"}, {"Host": "evil.example"}, {"Content-Type": "text/plain"}])
def test_decision_route_requires_auth_and_browser_guards(control, extra):
    port, calls = control
    headers = {"Content-Type": "application/json"}
    if extra:
        headers.update(auth.control_headers(TOKEN))
        headers.update(extra)
    code, _ = _call(port, "/decision", method="POST", headers=headers, body=b"{}")
    assert code == 403 and calls == []


def test_authenticated_decision_route_reaches_inference(control):
    port, calls = control
    code, body = _call(port, "/decision", method="POST", headers={**auth.control_headers(TOKEN), "Content-Type": "application/json"}, body=b"{}")
    assert code == 200 and body["answers"]["answer"]["noul"] == 0.9
    assert calls == ["decision"]


def test_empty_expected_token_refuses_everything():
    assert auth.request_allowed({"Host": "127.0.0.1:1", auth.TOKEN_HEADER: ""}, "", 1) is not None


def test_auth_file_is_owner_only(tmp_path):
    path = auth.write_auth_file(tmp_path / "run", "tok", "key")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    assert auth.read_auth_file(path.parent) == ("tok", "key")
    assert auth.read_auth_file(tmp_path / "missing") is None


@pytest.mark.parametrize("name", [
    "../../../../etc/x.gguf", "a/b.gguf", "..gguf", ".hidden.gguf", "model.bin", "",
    "m\x00.gguf", " model.gguf", "a\\b.gguf", "x/../y.gguf",
])
def test_unsafe_filenames_are_rejected(name):
    assert sv._safe_filename(name) is False


def test_plain_gguf_filename_is_accepted():
    assert sv._safe_filename("Qwen3.8-27B-UD-Q4_K_XL.gguf") is True


def _bare_supervisor(tmp_path):
    sup = object.__new__(sv.Supervisor)
    sup.models_dir = tmp_path
    sup.lock = threading.Lock()
    return sup


def test_select_refuses_traversal_before_touching_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "fetch_quants", lambda repo: pytest.fail("must not list for an unsafe name"))
    sup = _bare_supervisor(tmp_path)
    res = sup.select(sv.MODEL_GROUPS[0].key, "x/../../../../tmp/owned.gguf")
    assert res == {"error": "invalid filename"}
    assert list(tmp_path.iterdir()) == []


def test_select_refuses_a_file_the_group_does_not_list(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "fetch_quants", lambda repo: [types.SimpleNamespace(filename="listed.gguf")])
    sup = _bare_supervisor(tmp_path)
    res = sup.select(sv.MODEL_GROUPS[0].key, "not-listed.gguf")
    assert "is not a file of" in res["error"]
    assert not sup.lock.locked()


def test_server_command_carries_the_session_key(monkeypatch, tmp_path):
    from localcode.ui import server_cmd
    monkeypatch.setenv(auth.KEY_ENV, "k" * 43)
    cmd = server_cmd.server_command(str(tmp_path / "m.gguf"), 8123, "m")
    assert cmd[cmd.index("--api-key") + 1] == "k" * 43
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    monkeypatch.delenv(auth.KEY_ENV)
    assert "--api-key" not in server_cmd.server_command(str(tmp_path / "m.gguf"), 8123, "m")


def test_server_key_is_redacted_from_diagnostics():
    from localcode.ui.auth import redact_command
    command = ["server", "--api-key", "secret", "--port", "8081", "--api-key=other"]
    assert redact_command(command) == ["server", "--api-key", "[redacted]", "--port", "8081", "--api-key=[redacted]"]
    assert command[2] == "secret"
