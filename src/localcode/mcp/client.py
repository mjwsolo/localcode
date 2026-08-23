"""MCPClient + the process-wide registry of connected MCP servers.

`MCPClient` speaks one server connection through the official `mcp` SDK
(stdio / streamable-HTTP / SSE). The transport context manager and the
`ClientSession` are entered once inside a long-lived "runner" coroutine on
the shared event loop (anyio cancel scopes must be entered/exited on the
same task), and sync methods talk to the live session by submitting
coroutines to the loop.

The registry functions (`connect_all`/`list_connected`/`get_client`/
`shutdown_all`, plus `call`/`mcp_tool_schemas`/`dispatch_mcp_tool`) manage
a process-wide dict of connected clients.
"""
from __future__ import annotations

import asyncio
import concurrent.futures

# ── Official MCP SDK imports ─────────────────────────────────────────────
from mcp import ClientSession

from ._bridge import _CONNECT_TIMEOUT, _DEFAULT_TIMEOUT, _EventLoopThread
from ._config import load_mcp_config
from ._transports import build_transport


# ── Client ───────────────────────────────────────────────────────────────
class MCPClient:
    """One MCP server connection, spoken through the official SDK.

    Supports stdio, streamable-HTTP, and SSE transports. The transport
    context manager and the `ClientSession` are entered once inside a
    long-lived "runner" coroutine on the shared event loop (anyio cancel
    scopes must be entered/exited on the same task, so we keep that one task
    alive until close()). Sync methods talk to the live session by
    submitting coroutines to the loop.
    """

    def __init__(
        self,
        name: str,
        *,
        transport: str = "stdio",
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str = "",
        headers: dict[str, str] | None = None,
        oauth: bool = False,
    ) -> None:
        self.name = name
        self.transport = (transport or "stdio").lower()
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.url = url
        self.headers = headers or {}
        self.oauth = oauth

        self._bridge = _EventLoopThread.instance()
        self._session: ClientSession | None = None
        self._runner_future: concurrent.futures.Future | None = None
        self._ready: concurrent.futures.Future | None = None
        self._stop_event: asyncio.Event | None = None
        self._closed = False
        self._post_error: Exception | None = None

    # -- transport factory -------------------------------------------------
    def _make_transport(self):
        """Return the async context manager for the configured transport."""
        return build_transport(
            name=self.name,
            transport=self.transport,
            command=self.command,
            args=self.args,
            env=self.env,
            url=self.url,
            headers=self.headers,
            oauth=self.oauth,
        )

    # -- runner coroutine --------------------------------------------------
    async def _runner(self) -> None:
        """Open transport + session, signal ready, then idle until stop.

        Everything (enter and exit of the async-with blocks) happens inside
        this single task so anyio's cancel scopes stay task-bound.
        """
        self._stop_event = asyncio.Event()
        try:
            async with self._make_transport() as streams:
                # stdio/sse yield (read, write); streamable-http yields
                # (read, write, get_session_id) — take the first two.
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    if self._ready is not None and not self._ready.done():
                        self._ready.set_result(True)
                    await self._stop_event.wait()
        except Exception as exc:  # noqa: BLE001 - surface to connecting thread
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(exc)
            else:
                self._post_error = exc
        finally:
            self._session = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self, timeout: float = _CONNECT_TIMEOUT) -> None:
        """Start the runner and block until the session is initialized."""
        if self._session is not None:
            return
        self._ready = concurrent.futures.Future()
        self._runner_future = self._bridge.submit(self._runner())
        try:
            self._ready.result(timeout=timeout)
        except Exception as exc:
            # Make sure a half-started runner gets cleaned up.
            self.close()
            raise RuntimeError(
                f"MCP server {self.name!r} failed to connect: {exc}"
            ) from exc

    def initialize(self) -> None:
        """Back-compat alias — connect if not already connected."""
        self.connect()

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(f"MCP server {self.name!r} is not connected")
        return self._session

    # -- public sync API ---------------------------------------------------
    def list_tools(self) -> list[dict]:
        """Return the server's tools as plain dicts with `name`,
        `description`, and `inputSchema` (JSONSchema)."""
        if self._session is None:
            self.connect()
        session = self._require_session()
        result = self._bridge.run(session.list_tools(), timeout=_DEFAULT_TIMEOUT)
        tools: list[dict] = []
        for tool in result.tools:
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema
                    or {"type": "object", "properties": {}},
                }
            )
        return tools

    def call_tool(self, name: str, arguments: dict) -> str:
        """Invoke a tool; return the concatenated text content blocks."""
        if self._session is None:
            self.connect()
        session = self._require_session()
        result = self._bridge.run(
            session.call_tool(name, arguments or {}),
            timeout=_DEFAULT_TIMEOUT,
        )
        out: list[str] = []
        for block in result.content:
            # TextContent blocks have type == "text" and a .text attr.
            text = getattr(block, "text", None)
            if text is not None and getattr(block, "type", None) == "text":
                out.append(text)
        joined = "\n".join(out)
        if not joined:
            return "(MCP tool returned no text content)"
        if result.isError:
            return f"MCP tool {name!r} reported an error:\n{joined}"
        return joined

    def health(self) -> tuple[bool, str]:
        """Lightweight liveness check used by Toolkit.diagnostics()."""
        if self._closed:
            return False, "closed"
        if self._post_error is not None:
            return False, f"errored ({self._post_error})"
        if self._session is None:
            return False, "not connected"
        if self._runner_future is not None and self._runner_future.done():
            return False, "runner exited"
        return True, f"running ({self.transport})"

    def close(self) -> None:
        """Tear down the session + transport on the shared loop."""
        if self._closed:
            return
        self._closed = True
        # Signal the runner to leave its idle wait so the async-with blocks
        # exit cleanly on their own task.
        if self._stop_event is not None:
            try:
                self._bridge.loop.call_soon_threadsafe(self._stop_event.set)
            except Exception:
                pass
        if self._runner_future is not None:
            try:
                self._runner_future.result(timeout=10)
            except Exception:
                # Best-effort: cancel if it didn't exit in time.
                try:
                    self._runner_future.cancel()
                except Exception:
                    pass
        self._session = None


# ── Process-wide registry ─────────────────────────────────────────────────
# Registry of connected MCP clients (name → MCPClient).
_clients: dict[str, MCPClient] = {}

# Last connection error per server name, populated by connect_all(). Lets the
# TUI show WHY a configured server isn't connected — the error text is otherwise
# only returned transiently from connect_all() and then lost. Cleared for a
# server the moment it connects successfully.
_last_errors: dict[str, str] = {}


def _client_from_config(name: str, cfg: dict) -> MCPClient:
    """Build an MCPClient from one server's config dict (transport-aware)."""
    transport = (cfg.get("transport") or "stdio").lower()
    return MCPClient(
        name=name,
        transport=transport,
        command=cfg.get("command", "") or "",
        args=cfg.get("args", []) or [],
        env=cfg.get("env", {}) or {},
        url=cfg.get("url", "") or "",
        headers=cfg.get("headers", {}) or {},
        oauth=bool(cfg.get("oauth", False)),
    )


def connect_all() -> tuple[int, list[str]]:
    """Connect every configured MCP server. Returns (count_ok, errors)."""
    errors: list[str] = []
    config = load_mcp_config()
    for name, server_cfg in config.items():
        if name in _clients:
            continue  # already connected
        try:
            cli = _client_from_config(name, server_cfg or {})
            cli.connect()
            _clients[name] = cli
            _last_errors.pop(name, None)  # cleared on a successful connect
        except Exception as e:
            errors.append(f"{name}: {e}")
            _last_errors[name] = str(e)
    return len(_clients), errors


def get_client(name: str) -> "MCPClient | None":
    """Return the connected MCPClient for `name`, or None if not connected."""
    return _clients.get(name)


def list_connected() -> list[tuple[str, list[dict]]]:
    """Return [(server_name, [tools])] for every connected MCP server."""
    out = []
    for name, cli in _clients.items():
        try:
            out.append((name, cli.list_tools()))
        except Exception:
            out.append((name, []))
    return out


def server_status() -> list[dict]:
    """Return one honest status row per CONFIGURED server, for the TUI.

    Combines the config (every server the user wrote in mcp.json) with the live
    registry (which of them actually connected) and `_last_errors` (why the
    others didn't). Each row is a plain dict:

        {
          "name": str,
          "transport": str,        # "stdio" | "http" | "sse"
          "oauth": bool,           # http/sse auth flag from config (else False)
          "connected": bool,       # is there a live, healthy session?
          "tools": list[dict],     # the server's tools ([] unless connected)
          "error": str | None,     # last connect error, if it failed
        }

    Only reports what the client can actually tell us — connected / failed /
    not-connected. It does NOT invent auth states a stdio server has no way to
    report; `oauth` is surfaced only because it comes straight from the config.
    """
    config = load_mcp_config()
    rows: list[dict] = []
    for name, cfg in config.items():
        cfg = cfg or {}
        cli = _clients.get(name)
        connected = False
        tools: list[dict] = []
        if cli is not None:
            ok, _ = cli.health()
            connected = ok
            if ok:
                try:
                    tools = cli.list_tools()
                except Exception as e:  # noqa: BLE001
                    connected = False
                    _last_errors[name] = str(e)
        error = None if connected else _last_errors.get(name)
        rows.append(
            {
                "name": name,
                "transport": (cfg.get("transport") or "stdio").lower(),
                "oauth": bool(cfg.get("oauth", False)),
                "connected": connected,
                "tools": tools,
                "error": error,
            }
        )
    return rows


def call(server: str, tool_name: str, arguments: dict) -> str:
    cli = _clients.get(server)
    if cli is None:
        return f"REJECTED: MCP server {server!r} is not connected."
    try:
        return cli.call_tool(tool_name, arguments)
    except Exception as e:
        return f"MCP call {server}.{tool_name} failed: {e}"


def disconnect(name: str) -> bool:
    """Tear down ONE connected server's session and drop it from the registry.

    Returns True if a live client was closed, False if the server wasn't
    connected. The server stays in mcp.json — `r` (connect_all) reconnects it —
    so this is a runtime enable/disable, not a config edit. Reversible and
    honest: it only does what the client can actually do.
    """
    cli = _clients.pop(name, None)
    if cli is None:
        return False
    try:
        cli.close()
    except Exception:
        pass
    return True


def shutdown_all() -> None:
    """Tear down all MCP connections on app exit."""
    for cli in _clients.values():
        try:
            cli.close()
        except Exception:
            pass
    _clients.clear()
    # A fresh reload should re-derive every server's status from scratch, so a
    # stale error from a prior config doesn't linger after shutdown.
    _last_errors.clear()


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
