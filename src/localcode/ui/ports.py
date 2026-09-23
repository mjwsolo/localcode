"""Reject an existing localcode instance, then choose unused local ports."""
from __future__ import annotations

import json
import socket
import urllib.request


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
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=0.3) as response:
                status = json.load(response)
                int(status["port"])
                return port, status
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def choose_ports(model_ports=range(8123, 8200), control_ports=range(8323, 8400)) -> tuple[int, int]:
    control = None
    for port in control_ports:
        if available(port):
            if control is None:
                control = port
            continue
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=0.3) as response:
                int(json.load(response)["port"])
                raise RuntimeError("localcode is already running in another terminal; a second window attaches to its model server.")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    model = next((port for port in model_ports if available(port)), None)
    if model is None or control is None:
        raise RuntimeError("No free local model/control port pair")
    return model, control


if __name__ == "__main__":
    try:
        print(*choose_ports())
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(1)
