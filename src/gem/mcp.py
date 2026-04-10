from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from queue import Queue, Empty
import subprocess
import threading
import time
from typing import Any

from .config import ensure_home_dirs


@dataclass(slots=True)
class McpServerConfig:
    name: str
    command: str
    args: list[str]


def mcp_config_path() -> Path:
    return ensure_home_dirs() / "mcp.json"


def load_mcp_configs() -> list[McpServerConfig]:
    path = mcp_config_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [McpServerConfig(name=item["name"], command=item["command"], args=list(item.get("args", []))) for item in data]


def save_mcp_configs(configs: list[McpServerConfig]) -> Path:
    path = mcp_config_path()
    payload = [{"name": cfg.name, "command": cfg.command, "args": cfg.args} for cfg in configs]
    path.write_text(json.dumps(payload, indent=2))
    return path


def add_mcp_config(name: str, command: str, args: list[str]) -> Path:
    configs = [cfg for cfg in load_mcp_configs() if cfg.name != name]
    configs.append(McpServerConfig(name=name, command=command, args=args))
    return save_mcp_configs(configs)


class McpStdioClient:
    def __init__(self, config: McpServerConfig, timeout_seconds: float = 5.0) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.proc = subprocess.Popen(
            [config.command, *config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._request_id = 0
        self._stdout_queue: Queue[dict[str, Any]] = Queue()
        self._stderr_queue: Queue[str] = Queue()
        self._stdout_thread = threading.Thread(target=self._capture_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._capture_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _capture_stdout(self) -> None:
        if self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._stdout_queue.put(json.loads(line))
            except Exception:
                continue

    def _capture_stderr(self) -> None:
        if self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            self._stderr_queue.put(line.rstrip("\n"))

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                payload["params"] = params
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            deadline = time.time() + self.timeout_seconds
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise RuntimeError(
                        f"MCP server {self.config.name} timed out during {method}; stderr={self.recent_stderr() or 'none'}"
                    )
                try:
                    data = self._stdout_queue.get(timeout=remaining)
                except Empty as exc:
                    raise RuntimeError(
                        f"MCP server {self.config.name} timed out during {method}; stderr={self.recent_stderr() or 'none'}"
                    ) from exc
                if data.get("id") == req_id:
                    if "error" in data:
                        raise RuntimeError(str(data["error"]))
                    return data.get("result")

    def initialize(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "gem", "version": "0.0.1"},
            },
        )
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.proc.stdin.flush()

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list", {})
        return list(result.get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        return "\n".join(item.get("text", "") for item in content if item.get("type") == "text") or json.dumps(result)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()

    def health(self) -> tuple[bool, str]:
        if self.proc.poll() is not None:
            return False, f"process exited with {self.proc.returncode}; stderr={self.recent_stderr() or 'none'}"
        return True, "running"

    def recent_stderr(self, limit: int = 8) -> str:
        items: list[str] = []
        try:
            while True:
                items.append(self._stderr_queue.get_nowait())
        except Empty:
            pass
        return " | ".join(items[-limit:])
