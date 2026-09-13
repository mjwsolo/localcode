"""Choose local ports without sharing an idle supervisor's future model server."""
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


def choose_ports(model_ports=range(8123, 8200), control_ports=range(8323, 8400)) -> tuple[int, int]:
    reserved: set[int] = set()
    control = None
    for port in control_ports:
        if available(port):
            if control is None:
                control = port
            continue
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=0.3) as response:
                reserved.add(int(json.load(response)["port"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    model = next((port for port in model_ports if port not in reserved and available(port)), None)
    if model is None or control is None:
        raise RuntimeError("No free local model/control port pair")
    return model, control


if __name__ == "__main__":
    print(*choose_ports())
