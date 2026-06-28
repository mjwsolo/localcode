"""Transport construction + best-effort OAuth for the MCP package.

`build_transport` returns the async context manager for a server's
configured transport (stdio / streamable-HTTP / SSE). OAuth is a
best-effort construction only: LocalCode runs headless, so the
authorization-code callback degrades with a clear error instead of
hanging — prefer a pre-minted token via `headers`.
"""
from __future__ import annotations

import os

# ── Official MCP SDK imports ─────────────────────────────────────────────
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client


class _InMemoryTokenStorage:
    """A TokenStorage that keeps tokens/registration in process memory.

    Tokens are NOT persisted across runs, so every process start triggers a
    fresh OAuth flow. Good enough for a best-effort implementation; a real
    deployment would persist these under ~/.localcode.
    """

    def __init__(self) -> None:
        self._tokens = None
        self._client_info = None

    async def get_tokens(self):
        return self._tokens

    async def set_tokens(self, tokens) -> None:
        self._tokens = tokens

    async def get_client_info(self):
        return self._client_info

    async def set_client_info(self, client_info) -> None:
        self._client_info = client_info


def _build_oauth_provider(url: str):
    """Build an SDK OAuthClientProvider for `url`, best-effort.

    LIMITATION: a complete OAuth authorization-code flow needs an interactive
    browser + a loopback redirect listener to capture the `code`. LocalCode
    runs headless (CLI/agent), so we cannot drive that flow automatically.
    We construct a real provider so the auth *config path* is exercised and
    so a future interactive frontend could reuse it, but the callback handler
    degrades with a clear, actionable error instead of hanging. For servers
    that need OAuth, prefer passing a pre-minted token via `headers` (e.g.
    {"Authorization": "Bearer ..."}) until interactive OAuth is wired up.
    """
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    async def _redirect_handler(authorization_url: str) -> None:
        # In an interactive frontend this would open a browser. Headless: noop.
        # (The error is raised from the callback handler below.)
        return None

    async def _callback_handler() -> tuple[str, str | None]:
        raise RuntimeError(
            "OAuth for MCP requires an interactive browser flow, which is not "
            "available in headless LocalCode. Configure the server with a "
            "pre-minted token via `headers` (e.g. Authorization: Bearer ...) "
            "instead of `oauth: true`."
        )

    return OAuthClientProvider(
        server_url=url,
        client_metadata=OAuthClientMetadata(
            client_name="LocalCode",
            redirect_uris=["http://localhost:8765/callback"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=_InMemoryTokenStorage(),
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
    )


def build_transport(
    *,
    name: str,
    transport: str,
    command: str,
    args: list[str],
    env: dict[str, str],
    url: str,
    headers: dict[str, str],
    oauth: bool,
):
    """Return the async context manager for the configured transport."""
    if transport in ("stdio", ""):
        if not command:
            raise RuntimeError(
                f"MCP server {name!r}: stdio transport needs a 'command'"
            )
        full_env = os.environ.copy()
        full_env.update(env)
        params = StdioServerParameters(command=command, args=args, env=full_env)
        return stdio_client(params)
    if transport in ("http", "streamable-http", "streamable_http"):
        if not url:
            raise RuntimeError(
                f"MCP server {name!r}: http transport needs a 'url'"
            )
        auth = _build_oauth_provider(url) if oauth else None
        return streamablehttp_client(url, headers=headers or None, auth=auth)
    if transport == "sse":
        if not url:
            raise RuntimeError(
                f"MCP server {name!r}: sse transport needs a 'url'"
            )
        auth = _build_oauth_provider(url) if oauth else None
        return sse_client(url, headers=headers or None, auth=auth)
    raise RuntimeError(
        f"MCP server {name!r}: unknown transport {transport!r} "
        f"(use 'stdio', 'http', or 'sse')"
    )
