"""Real TCP coverage for port handover; no model weights required."""
import http.client
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localcode.ui.gateway import ModelGateway, PortChangeError


def test_live_port_handover_preserves_stream_and_authentication(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path))
    finish = threading.Event()

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer test-private-key":
                self.send_error(401)
                return
            self.send_response(200)
            self.end_headers()
            if self.path == "/stream":
                self.wfile.write(b"data: one\n\n")
                self.wfile.flush()
                finish.wait(5)
                self.wfile.write(b"data: two\n\n")
            else:
                self.wfile.write(b'{"data":[]}')

    try:
        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    except PermissionError:
        pytest.skip("Sandbox disallows binding a loopback listener; run on CI or a local terminal")
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    gateway = None
    stream = None
    try:
        gateway = ModelGateway(backend.server_port, 0, 1)
        previous = gateway.port
        stream = http.client.HTTPConnection("127.0.0.1", previous, timeout=3)
        stream.request("GET", "/stream", headers={"Authorization": "Bearer test-private-key"})
        response = stream.getresponse()
        assert response.status == 200
        assert response.readline() == b"data: one\n"
        assert response.readline() == b"\n"
        result = gateway.change(None)
        assert result["port"] != previous and result["restart_required"] is False
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", previous), timeout=1)
        finish.set()
        assert response.read() == b"data: two\n\n"  # old TCP stream survives retirement
        for headers, expected in (({}, 401), ({"Authorization": "Bearer test-private-key"}, 200)):
            client = http.client.HTTPConnection("127.0.0.1", gateway.port, timeout=3)
            try:
                client.request("GET", "/v1/models", headers=headers)
                reply = client.getresponse()
                assert reply.status == expected
                reply.read()
            finally:
                client.close()
        current = gateway.port
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            with pytest.raises(PortChangeError, match="current port is unchanged"):
                gateway.change(occupied.getsockname()[1])
        assert gateway.port == current
    finally:
        finish.set()
        if stream is not None:
            stream.close()
        if gateway is not None:
            gateway.close()
        backend.shutdown()
        backend.server_close()
