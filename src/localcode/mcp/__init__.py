"""MCP (Model Context Protocol) client + tool-registration.

Lets a user add MCP servers to LocalCode in `~/.localcode/mcp.json`.
Each server's tools get auto-registered with the agent so the model
can call them like any other tool.

This package is built on the **official `mcp` Python SDK** (the
`modelcontextprotocol` project). The SDK is fully async (anyio), but
LocalCode's toolkit/agent are synchronous, so this package owns an
async↔sync bridge: a single dedicated background thread runs a
persistent asyncio event loop, and every public method submits a
coroutine to that loop via `asyncio.run_coroutine_threadsafe(...)`.
Each server's transport context + `ClientSession` are opened once and
kept alive on that loop for the process lifetime; they're torn down in
`shutdown_all()` / `MCPClient.close()`.

Config shape — `~/.localcode/mcp.json`, key `mcpServers`, per server:

    {
      "mcpServers": {
        "filesystem": {                       # stdio (default transport)
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me"],
          "env": {}
        },
        "remote-http": {                      # streamable HTTP
          "transport": "http",
          "url": "https://example.com/mcp",
          "headers": {"Authorization": "Bearer ..."},
          "oauth": true                       # optional, http/sse only
        },
        "remote-sse": {                       # legacy SSE
          "transport": "sse",
          "url": "https://example.com/sse",
          "headers": {}
        }
      }
    }

Transport is chosen by the `transport` field; it defaults to "stdio" so
existing stdio configs (no `transport` key) keep working unchanged.

We use `initialize`, `tools/list`, and `tools/call`. No prompts,
resources, sampling, or notifications — good enough for the common
"let the agent use my MCP server" case.

This is a package; the implementation is split across `_bridge`,
`_transports`, `_config`, and `client` submodules. The full public API
is re-exported here so `from localcode.mcp import ...` is unchanged.
"""
from __future__ import annotations

from ._bridge import _CONNECT_TIMEOUT, _DEFAULT_TIMEOUT, _EventLoopThread
from ._config import MCP_CONFIG_PATH, _config_path, load_mcp_config
from ._transports import _InMemoryTokenStorage, _build_oauth_provider
from .client import (
    MCPClient,
    _client_from_config,
    _clients,
    call,
    connect_all,
    dispatch_mcp_tool,
    get_client,
    list_connected,
    mcp_tool_schemas,
    shutdown_all,
)

__all__ = [
    # config
    "MCP_CONFIG_PATH",
    "load_mcp_config",
    # client
    "MCPClient",
    # registry
    "connect_all",
    "list_connected",
    "get_client",
    "call",
    "shutdown_all",
    # tool schemas / dispatch
    "mcp_tool_schemas",
    "dispatch_mcp_tool",
]
