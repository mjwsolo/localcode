"""Async↔sync bridge for the MCP package.

The official `mcp` SDK is fully async (anyio), but LocalCode's
toolkit/agent are synchronous. This module owns a single dedicated
background thread running a persistent asyncio event loop; every public
MCP method submits a coroutine to that loop via
`asyncio.run_coroutine_threadsafe(...)`. Each server's transport context
+ `ClientSession` are opened once and kept alive on that loop for the
process lifetime.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading


# Default time budget (seconds) for a single round-trip to an MCP server.
_DEFAULT_TIMEOUT = 30.0
# Time budget for the initial connect (spawning a subprocess / TCP + TLS +
# initialize handshake can be slow, especially for `npx`/`uvx` cold starts).
_CONNECT_TIMEOUT = 60.0


class _EventLoopThread:
    """A singleton background thread running a persistent asyncio loop.

    All SDK coroutines run on this one loop, so a `ClientSession` opened on
    it stays usable for the process lifetime. Sync callers submit work with
    `run(coro, timeout)`.
    """

    _instance: "_EventLoopThread | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="localcode-mcp-loop", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @classmethod
    def instance(cls) -> "_EventLoopThread":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def run(self, coro, timeout: float | None = None):
        """Submit a coroutine to the loop and block for its result."""
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)

    def submit(self, coro) -> concurrent.futures.Future:
        """Submit a coroutine without blocking; returns its Future."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)
