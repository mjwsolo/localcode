"""Server liveness and persistent external-client port preferences."""
import subprocess
import sys

import pytest

from localcode.ui import ports
from localcode.ui.supervisor import Supervisor


@pytest.fixture(autouse=True)
def isolated_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path))


@pytest.mark.parametrize("value", [True, "8123", 0, 1023, 65536, 1.5])
def test_invalid_port_never_saved(value, tmp_path):
    with pytest.raises(ValueError):
        ports.save_port(value)
    assert not (tmp_path / "server-port.json").exists()


def test_preference_is_private_and_resettable(tmp_path):
    ports.save_port(9321)
    assert ports.preferred_port() == 9321
    assert (tmp_path / "server-port.json").stat().st_mode & 0o777 == 0o600
    ports.save_port(None)
    assert ports.preferred_port() is None


def test_busy_preferred_port_fails_instead_of_silently_changing(monkeypatch):
    ports.save_port(9321)
    monkeypatch.setattr(ports, "available", lambda _: False)
    with pytest.raises(RuntimeError, match="in use"):
        ports.choose_ports()


def test_preferred_model_port_does_not_collide_with_control(monkeypatch):
    ports.save_port(8323)
    monkeypatch.setattr(ports, "available", lambda _: True)
    assert ports.choose_ports() == (8323, 8324)


def test_corrupt_preference_can_be_replaced(tmp_path):
    (tmp_path / "server-port.json").write_text("broken")
    with pytest.raises(RuntimeError):
        ports.preferred_port()
    ports.save_port(None)
    assert ports.preferred_port() is None


def test_dead_or_unhealthy_model_never_reports_ready():
    sup = Supervisor.__new__(Supervisor)
    sup.state = {"state": "ready"}
    sup.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        sup.healthy = lambda: True
        assert sup.server_status()["server_ready"] is True
        sup.healthy = lambda: False
        assert sup.server_status()["state"] == "error"
        sup.proc.terminate()
        sup.proc.wait(timeout=5)
        sup.healthy = lambda: True
        assert sup.server_status()["server_running"] is False
        assert sup.server_status()["server_ready"] is False
        assert sup.state == {"state": "ready"}  # observation does not race model-switch state
    finally:
        if sup.proc.poll() is None:
            sup.proc.kill()
            sup.proc.wait()


def control_request(path, *, method="GET", token="test-token", body=None, gateway=None):
    """Exercise the real HTTP parser and auth checks without binding a socket."""
    import io
    import json
    from types import SimpleNamespace
    from localcode.ui.supervisor import make_handler

    sup = SimpleNamespace(gateway=gateway, control_token="test-token", control_port=8323, port=8123,
                          current="test-model", server_status=lambda: {"state": "ready", "server_ready": True})
    payload = json.dumps(body).encode() if body is not None else b""
    wire = (f"{method} {path} HTTP/1.0\r\nHost: 127.0.0.1:8323\r\n"
            f"X-Localcode-Token: {token}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload

    class Connection:
        output = b""
        def makefile(self, *_args):
            return io.BytesIO(wire)
        def sendall(self, data):
            self.output += bytes(data)

    connection = Connection()
    make_handler(sup)(connection, ("127.0.0.1", 1234), None)
    headers, result = connection.output.split(b"\r\n\r\n", 1)
    return int(headers.split()[1]), json.loads(result)


def test_server_details_are_authenticated_and_do_not_reveal_keys():
    code, data = control_request("/server", token="wrong")
    assert code == 403
    code, data = control_request("/server")
    assert code == 200
    assert data["base_url"] == "http://127.0.0.1:8123/v1"
    assert "api_key" not in data and "control_token" not in data


def test_port_route_requires_auth_and_validates_before_persisting(managed_gateway):
    code, _ = control_request("/server/port", gateway=managed_gateway, method="POST", token="wrong", body={"port": 9234})
    assert code == 403 and ports.preferred_port() is None
    for body in ([], {}, {"port": True}, {"port": 65536}):
        code, _ = control_request("/server/port", gateway=managed_gateway, method="POST", body=body)
        assert code == 400 and ports.preferred_port() is None
    code, data = control_request("/server/port", gateway=managed_gateway, method="POST", body={"port": 9234})
    assert code == 200 and data["restart_required"] is False
    assert ports.preferred_port() == 9234
    code, _ = control_request("/server/port", gateway=managed_gateway, method="POST", body={"port": None})
    assert code == 200 and ports.preferred_port() is None


def test_cli_can_reset_port_without_running_server(monkeypatch, capsys):
    from localcode.ui import api
    monkeypatch.setattr(api, "find_running", lambda: None)
    assert api.main(["--port", "9234"]) == 0
    assert ports.preferred_port() == 9234
    assert "next launch" in capsys.readouterr().out
    assert api.main(["--port", "auto"]) == 0
    assert ports.preferred_port() is None
    assert api.main(["--port", "65536"]) == 1
    assert ports.preferred_port() is None


def test_binary_path_check_does_not_hide_personal_paths_on_toolchain_lines():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "scripts" / "check_ui_paths.py"
    spec = importlib.util.spec_from_file_location("check_ui_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.unexpected_paths(["/Users/runner/build /Users/alice/project/file.ts"]) == {"/Users/alice/project/file.ts"}
    assert not module.unexpected_paths(["/Users/name/My Project", "/Users/administrator/Library/Services/buildkite-agent/builds/bun/file.rs"])


@pytest.fixture
def managed_gateway(monkeypatch):
    from localcode.ui import gateway

    class Listener:
        instances = []
        def __init__(self, port, backend, capacity):
            if port == 9999:
                raise OSError(48, "Address already in use")
            self.server_address = ("127.0.0.1", port or 49123)
            self.retired = False
            self.instances.append(self)
        def start(self):
            return self
        def retire(self):
            self.retired = True

    monkeypatch.setattr(gateway, "Listener", Listener)
    return gateway.ModelGateway(51234, 8123, 8323)


def test_live_port_change_starts_new_listener_and_retires_old(managed_gateway):
    old = managed_gateway.listener
    result = managed_gateway.change(9234)
    assert result["port"] == 9234 and result["restart_required"] is False
    assert old.retired and not managed_gateway.listener.retired
    assert managed_gateway.backend == 51234
    assert ports.preferred_port() == 9234


@pytest.mark.parametrize("port", [True, "8123", 0, 65536, 8323, 51234, 9999])
def test_rejected_port_keeps_old_listener_and_preference(managed_gateway, port):
    from localcode.ui.gateway import PortChangeError
    old = managed_gateway.listener
    with pytest.raises(PortChangeError):
        managed_gateway.change(port)
    assert managed_gateway.listener is old and not old.retired
    assert ports.preferred_port() is None


def test_save_failure_closes_new_listener_and_keeps_old(managed_gateway, monkeypatch):
    from localcode.ui import gateway
    old = managed_gateway.listener
    def fail(_):
        raise OSError("disk full")
    monkeypatch.setattr(gateway, "save_port", fail)
    with pytest.raises(gateway.PortChangeError, match="current port is unchanged"):
        managed_gateway.change(9234)
    assert managed_gateway.listener is old and not old.retired
    assert gateway.Listener.instances[-1].retired
    assert not managed_gateway._lock.locked()


def test_concurrent_change_and_shutdown_are_explicit_errors(managed_gateway):
    from localcode.ui.gateway import PortChangeError
    managed_gateway._lock.acquire()
    try:
        with pytest.raises(PortChangeError, match="in progress"):
            managed_gateway.change(9234)
    finally:
        managed_gateway._lock.release()
    managed_gateway.close()
    managed_gateway.close()  # idempotent cleanup
    with pytest.raises(PortChangeError, match="shutting down"):
        managed_gateway.change(9234)


def test_same_port_is_idempotent_and_auto_allocates(managed_gateway):
    old = managed_gateway.listener
    assert managed_gateway.change(8123)["port"] == 8123
    assert managed_gateway.listener is old and not old.retired
    assert managed_gateway.change(None)["port"] == 49123
    assert old.retired and ports.preferred_port() is None


def test_tcp_relay_preserves_streaming_and_half_close():
    import socket
    import threading
    from localcode.ui.gateway import relay
    client, front = socket.socketpair()
    back, model = socket.socketpair()
    failures = []
    def run():
        try:
            relay(front, back, idle_timeout=2)
        except Exception as exc:
            failures.append(exc)
        finally:
            front.close()
            back.close()
    worker = threading.Thread(target=run)
    worker.start()
    try:
        client.settimeout(2)
        model.settimeout(2)
        request = b"POST /v1/chat/completions HTTP/1.1\r\nAuthorization: Bearer private-key\r\n\r\n"
        client.sendall(request)
        assert model.recv(4096) == request
        client.shutdown(socket.SHUT_WR)
        assert model.recv(1) == b""
        for chunk in (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n", b"data: first\n\n", b"data: [DONE]\n\n"):
            model.sendall(chunk)
            assert client.recv(4096) == chunk  # no waiting for the whole response
        model.shutdown(socket.SHUT_WR)
        assert client.recv(1) == b""
    finally:
        client.close()
        model.close()
        worker.join(timeout=3)
    assert not worker.is_alive() and not failures


def test_tcp_relay_has_idle_timeout():
    import socket
    from localcode.ui.gateway import relay
    client, front = socket.socketpair()
    back, model = socket.socketpair()
    try:
        with pytest.raises(TimeoutError):
            relay(front, back, idle_timeout=0.02)
    finally:
        for sock in (client, front, back, model):
            sock.close()


def test_cli_changes_live_port_through_authenticated_control(monkeypatch, capsys):
    import io
    import json
    from localcode.ui import api
    monkeypatch.setattr(api, "find_running", lambda: (8323, {"port": 8123}))
    monkeypatch.setattr(api, "read_auth_file", lambda _: ("control-key", "inference-key"))
    requests = []
    def request(req, timeout):
        requests.append(req)
        return io.BytesIO(b'{"ok":true,"base_url":"http://127.0.0.1:9234/v1"}')
    monkeypatch.setattr(api.urllib.request, "urlopen", request)
    assert api.main(["--port", "9234"]) == 0
    assert requests[0].full_url == "http://127.0.0.1:8323/server/port"
    assert requests[0].get_header("X-localcode-token") == "control-key"
    assert json.loads(requests[0].data) == {"port": 9234}
    assert "9234/v1" in capsys.readouterr().out
    assert ports.preferred_port() is None  # only the supervisor commits this


def test_cli_reports_live_rejection_without_saving_offline(monkeypatch, capsys):
    import io
    from urllib.error import HTTPError
    from localcode.ui import api
    monkeypatch.setattr(api, "find_running", lambda: (8323, {"port": 8123}))
    monkeypatch.setattr(api, "read_auth_file", lambda _: ("control-key", "inference-key"))
    def fail(req, timeout):
        raise HTTPError(req.full_url, 409, "Conflict", {}, io.BytesIO(b'{"error":"port occupied; unchanged"}'))
    monkeypatch.setattr(api.urllib.request, "urlopen", fail)
    assert api.main(["--port", "9234"]) == 1
    assert "port occupied" in capsys.readouterr().err
    assert ports.preferred_port() is None


def test_port_route_reports_conflict_and_current_endpoint(managed_gateway):
    code, data = control_request("/server/port", method="POST", body={"port": 9999}, gateway=managed_gateway)
    assert code == 409 and "unchanged" in data["error"]
    code, data = control_request("/server", gateway=managed_gateway)
    assert code == 200 and data["port"] == 8123


def test_port_route_reports_older_supervisor_explicitly():
    code, data = control_request("/server/port", method="POST", body={"port": 9234})
    assert code == 409 and "Restart LocalCode once" in data["error"]


def test_gateway_unavailable_response_is_valid_http():
    import json
    import socket
    from localcode.ui.gateway import unavailable
    client, server = socket.socketpair()
    try:
        unavailable(server)
        response = client.recv(4096)
        headers, body = response.split(b"\r\n\r\n", 1)
        assert headers.startswith(b"HTTP/1.1 503 Service Unavailable")
        assert f"Content-Length: {len(body)}".encode() in headers
        assert "unavailable" in json.loads(body)["error"]["message"]
    finally:
        client.close()
        server.close()
