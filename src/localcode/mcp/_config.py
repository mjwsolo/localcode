"""Config loading for the MCP package.

Reads `~/.localcode/mcp.json` (key `mcpServers`), honoring LOCALCODE_HOME
like the rest of the app.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


MCP_CONFIG_PATH = Path.home() / ".localcode" / "mcp.json"


def _config_path() -> Path:
    """Resolve the mcp.json path, honoring LOCALCODE_HOME like the rest of
    the app (config.get_home_dir). Falls back to ~/.localcode."""
    override = os.environ.get("LOCALCODE_HOME")
    if override:
        return Path(override).expanduser() / "mcp.json"
    return MCP_CONFIG_PATH


def load_mcp_config() -> dict[str, dict]:
    """Read ~/.localcode/mcp.json. Returns the `mcpServers` dict."""
    path = _config_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text()).get("mcpServers", {})
    except Exception:
        return {}


def _read_full_config(path: Path) -> dict:
    """The whole mcp.json document (not just `mcpServers`), so a write preserves
    any other keys the user keeps in the file. Empty dict if missing/invalid."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def add_mcp_server(name: str, entry: dict) -> Path:
    """Add (or replace) one server under `mcpServers` and write mcp.json back,
    creating the file and its parent directory if needed. Returns the path.

    This is what the in-TUI "add server" form calls so a user never has to
    hand-edit the JSON. The written shape is exactly what ``load_mcp_config``
    reads and ``connect_all`` consumes.
    """
    path = _config_path()
    data = _read_full_config(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[name] = entry
    data["mcpServers"] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def remove_mcp_server(name: str) -> bool:
    """Drop one server from mcp.json. True if it was there, False otherwise."""
    path = _config_path()
    data = _read_full_config(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or name not in servers:
        return False
    del servers[name]
    data["mcpServers"] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return True
