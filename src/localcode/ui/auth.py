"""Per-session secrets for the two local listeners.

Both the model supervisor's control API and llama-server bind to loopback only,
but any web page in the user's browser can still send requests to localhost
ports. So every session carries two random secrets:

- the control token: required (as the X-Localcode-Token header) on every
  control-API request, reads included, since /status and /voice/stop leak on
  their own;
- the server key: llama-server's --api-key, sent as a bearer token by the UI
  runtime and by the supervisor's and warm-up's own requests.

The launcher generates them and hands them to the supervisor and the UI through
the environment. The supervisor also writes them, owner-readable only, into the
run directory, so a second `localcode` (which attaches to the running session)
and developer tooling can find them.
"""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

TOKEN_ENV = "LOCALCODE_CONTROL_TOKEN"
KEY_ENV = "LOCALCODE_SERVER_KEY"
TOKEN_HEADER = "X-Localcode-Token"
AUTH_FILE = "auth.json"


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def control_token() -> str:
    return os.environ.get(TOKEN_ENV, "")


def server_key() -> str:
    return os.environ.get(KEY_ENV, "")


def control_headers(token: str | None = None) -> dict[str, str]:
    return {TOKEN_HEADER: token if token is not None else control_token()}


def server_headers(key: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {key if key is not None else server_key()}"}


def write_auth_file(run_dir: Path, token: str, key: str) -> Path:
    """Persist the session secrets owner-readable only (0600, inside a 0700 dir)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(run_dir, 0o700)
    except OSError:
        pass
    path = run_dir / AUTH_FILE
    tmp = path.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"control_token": token, "server_key": key}, f)
    os.replace(tmp, path)
    return path


def read_auth_file(run_dir: Path) -> tuple[str, str] | None:
    try:
        data = json.loads((run_dir / AUTH_FILE).read_text())
        return str(data["control_token"]), str(data["server_key"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def request_allowed(headers, expected_token: str, port: int, *, want_json: bool = False) -> str | None:
    """Why a control-API request must be refused, or None when it may proceed.

    Checked in order: a browser-originated request (Origin header present),
    a Host that is not this loopback listener (DNS rebinding), a wrong or
    missing token, and, for state-changing requests, a non-JSON content type
    (a cross-site form or no-cors fetch cannot set one).
    """
    if headers.get("Origin") is not None:
        return "cross-origin request refused"
    host = (headers.get("Host") or "").strip().lower()
    if host not in {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}:
        return "unexpected Host"
    if not expected_token or not secrets.compare_digest(headers.get(TOKEN_HEADER) or "", expected_token):
        return "missing or wrong control token"
    if want_json:
        ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return "Content-Type must be application/json"
    return None


def redact_command(command: list[str]) -> list[str]:
    """Keep the server key out of diagnostic logs."""
    redacted = list(command)
    for i, value in enumerate(redacted):
        if value == "--api-key" and i + 1 < len(redacted):
            redacted[i + 1] = "[redacted]"
        elif value.startswith("--api-key="):
            redacted[i] = "--api-key=[redacted]"
    return redacted
