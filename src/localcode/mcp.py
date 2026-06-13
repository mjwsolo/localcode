"""MCP (Model Context Protocol) client + tool-registration.

Lets a user add MCP servers to LocalCode in `~/.localcode/mcp.json`.
Each server's tools get auto-registered with the agent so the model
can call them like any other tool.

Config shape (standard MCP server convention):

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me"],
          "env": {}
        },
        "github": {
          "command": "uvx",
          "args": ["mcp-server-github"],
          "env": {"GITHUB_TOKEN": "ghp_..."}
        }
      }
    }

For each server, we spawn the subprocess once at startup, speak the
MCP JSON-RPC protocol over its stdin/stdout, and expose each declared
tool as `mcp_<server>_<tool>` in the agent's tool list.

This is a minimal viable client — covers `initialize`, `tools/list`,
and `tools/call`. No prompts, resources, sampling, or notifications.
Good enough for the common "let the agent use my filesystem MCP" case.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any


MCP_CONFIG_PATH = Path.home() / ".localcode" / "mcp.json"


class MCPClient:
    """One stdio-spoken MCP server connection."""

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None):
        self.name = name
        self._proc: subprocess.Popen | None = None
        self._req_id = 0
        self._lock = threading.Lock()
        self._initialized = False
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        try:
            self._proc = subprocess.Popen(
                [command, *args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1, env=full_env,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"MCP server {name!r} command not found: {command}") from e

    def _next_id(self) -> int:
        with self._lock:
            self._req_id += 1
            return self._req_id

    def _send(self, method: str, params: dict | None = None) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError(f"MCP server {self.name!r} is not running")
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        with self._lock:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            # Read one line of JSON-RPC response. MCP servers may emit
            # notifications (no id) — skip those and keep reading.
            for _ in range(50):
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"MCP server {self.name!r} closed stdout")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" not in msg:
                    continue  # notification, not our response
                if msg.get("id") == req["id"]:
                    if "error" in msg:
                        raise RuntimeError(f"MCP {self.name}.{method}: {msg['error']}")
                    return msg.get("result", {})
            raise RuntimeError(f"MCP server {self.name!r} response timeout")

    def initialize(self) -> None:
        if self._initialized:
            return
        self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "localcode", "version": "0.2"},
        })
        self._initialized = True

    def list_tools(self) -> list[dict]:
        """Return the list of tools the server exposes — each has a
        `name`, `description`, and `inputSchema` (JSONSchema)."""
        self.initialize()
        result = self._send("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        """Invoke a tool by name with JSON-encoded arguments. Returns
        the text content of the response."""
        self.initialize()
        result = self._send("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        # MCP spec: content is a list of {type, text|data|...} blocks.
        # We concatenate text blocks; ignore image/blob for now.
        out = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
        return "\n".join(out) or "(MCP tool returned no text content)"

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass


# Process-wide registry of connected MCP clients (name → MCPClient).
_clients: dict[str, MCPClient] = {}


def load_mcp_config() -> dict[str, dict]:
    """Read ~/.localcode/mcp.json. Returns the `mcpServers` dict."""
    if not MCP_CONFIG_PATH.is_file():
        return {}
    try:
        return json.loads(MCP_CONFIG_PATH.read_text()).get("mcpServers", {})
    except Exception:
        return {}


def connect_all() -> tuple[int, list[str]]:
    """Spawn every configured MCP server. Returns (count_ok, errors)."""
    errors: list[str] = []
    config = load_mcp_config()
    for name, server_cfg in config.items():
        if name in _clients:
            continue  # already connected
        try:
            cli = MCPClient(
                name=name,
                command=server_cfg.get("command", ""),
                args=server_cfg.get("args", []) or [],
                env=server_cfg.get("env", {}) or {},
            )
            cli.initialize()
            _clients[name] = cli
        except Exception as e:
            errors.append(f"{name}: {e}")
    return len(_clients), errors


def list_connected() -> list[tuple[str, list[dict]]]:
    """Return [(server_name, [tools])] for every connected MCP server."""
    out = []
    for name, cli in _clients.items():
        try:
            out.append((name, cli.list_tools()))
        except Exception:
            out.append((name, []))
    return out


def call(server: str, tool_name: str, arguments: dict) -> str:
    cli = _clients.get(server)
    if cli is None:
        return f"REJECTED: MCP server {server!r} is not connected."
    try:
        return cli.call_tool(tool_name, arguments)
    except Exception as e:
        return f"MCP call {server}.{tool_name} failed: {e}"


def shutdown_all() -> None:
    """Tear down all MCP subprocesses on app exit."""
    for cli in _clients.values():
        try:
            cli.close()
        except Exception:
            pass
    _clients.clear()


def mcp_tool_schemas() -> list[dict]:
    """Return OpenAI-style tool schemas for every connected MCP
    server's tools. Each tool is renamed `mcp_<server>_<tool>` so
    multiple servers can expose tools with the same name without
    collision."""
    schemas = []
    for server, tools in list_connected():
        for t in tools:
            name = f"mcp_{server}_{t.get('name', '')}"
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        f"[MCP {server}] " + (t.get("description") or "")
                    )[:1000],
                    "parameters": t.get("inputSchema", {"type": "object"}),
                },
            })
    return schemas


def dispatch_mcp_tool(name: str, arguments: dict) -> str | None:
    """If `name` is `mcp_<server>_<tool>`, dispatch to that server.
    Returns None if not an MCP tool (so caller falls through to
    normal tool dispatch)."""
    if not name.startswith("mcp_"):
        return None
    rest = name[len("mcp_"):]
    if "_" not in rest:
        return None
    server, tool = rest.split("_", 1)
    return call(server, tool, arguments)
