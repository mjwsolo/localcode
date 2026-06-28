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
