from __future__ import annotations

import shutil
from typing import Any

from .config import AppConfig, save_config
from .mcp import add_mcp_config, load_mcp_configs


PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@latest"


def ensure_browser_mcp(config: AppConfig) -> str:
    path = add_mcp_config(
        config.browser.mcp_server_name,
        config.browser.launch_command,
        list(config.browser.launch_args or ["-y", PLAYWRIGHT_MCP_PACKAGE]),
    )
    save_config(config)
    return str(path)


def browser_status(config: AppConfig) -> list[str]:
    messages = [f"browser_enabled={config.browser.enabled}", f"mcp_server_name={config.browser.mcp_server_name}"]
    messages.append(f"launch={config.browser.launch_command} {' '.join(config.browser.launch_args or [])}".strip())
    messages.append("npx=present" if shutil.which("npx") else "npx=missing")
    configured = {cfg.name for cfg in load_mcp_configs()}
    messages.append("preset=configured" if config.browser.mcp_server_name in configured else "preset=missing")
    return messages


def find_browser_tool(tools: dict[str, Any], action: str) -> str | None:
    preferred = {
        "open": ("navigate", "goto", "open"),
        "snapshot": ("snapshot", "screenshot"),
    }
    wanted = preferred.get(action, (action,))
    for name in tools:
        if not name.startswith("mcp__"):
            continue
        parts = name.split("__")
        if len(parts) < 3:
            continue
        server_name = parts[1]
        if server_name != "browser":
            continue
        lowered = parts[2].lower()
        if any(token in lowered for token in wanted):
            return name
    return None
