"""Reject an existing localcode instance, then choose unused local ports."""
from __future__ import annotations

import json
import os
import socket
import tempfile
import urllib.request

from localcode.ui import run_dir
from localcode.ui.auth import control_headers, read_auth_file


def preferred_port() -> int | None:
    try:
        value = json.loads((run_dir() / "server-port.json").read_text())["port"]
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Invalid server port preference; reset it with localcode api --port auto") from exc
    if value is not None and (type(value) is not int or not 1024 <= value <= 65535):
        raise RuntimeError("Saved server port must be between 1024 and 65535")
    return value


def save_port(value: int | None) -> None:
    """Save a preference for the next launch; never interrupt a live server."""
    if value is not None and (type(value) is not int or not 1024 <= value <= 65535):
        raise ValueError("Port must be an integer between 1024 and 65535, or null for automatic")
    directory = run_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".server-port-", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"port": value}, stream)
        os.replace(name, directory / "server-port.json")
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _status(port: int) -> dict:
    """/status of a supervisor on this port, authenticated with the session
    token the supervisor left in the run directory."""
    auth = read_auth_file(run_dir())
    req = urllib.request.Request(f"http://127.0.0.1:{port}/status",
                                 headers=control_headers(auth[0] if auth else ""))
    with urllib.request.urlopen(req, timeout=1.5) as response:
        return json.load(response)


def available(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_running(control_ports=range(8323, 8400)) -> tuple[int, dict] | None:
    """The control port and /status of a supervisor already running for this user, if any."""
    for port in control_ports:
        if available(port):
            continue
        try:
            status = _status(port)
            int(status["port"])
            return port, status
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def choose_ports(model_ports=range(8123, 8200), control_ports=range(8323, 8400)) -> tuple[int, int]:
    preferred = preferred_port()
    if preferred is not None:
        if not available(preferred):
            raise RuntimeError(f"Configured model port {preferred} is in use; close that service or change server-port.json in {run_dir()}")
        model_ports = (preferred,)
    control = None
    for port in control_ports:
        if port == preferred:
            continue
        if available(port):
            if control is None:
                control = port
            continue
        try:
            int(_status(port)["port"])
            raise RuntimeError("localcode is already running in another terminal; a second window attaches to its model server.")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    model = next((port for port in model_ports if port != control and available(port)), None)
    if model is None or control is None:
        raise RuntimeError("No free local model/control port pair")
    return model, control


if __name__ == "__main__":
    try:
        print(*choose_ports())
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(1)
