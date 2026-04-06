"""Code intelligence — LSP client + offline fallback.

Tries real language servers first (pyright, typescript-language-server, gopls, rust-analyzer).
Falls back to offline tools (ruff, pyflakes, py_compile) when no server available.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Data types ──────────────────────────────────────────────────────

@dataclass
class Diagnostic:
    file: str
    line: int
    severity: str  # error | warning | info
    message: str
    source: str = ""

    def __str__(self) -> str:
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(self.severity, "?")
        return f"  {icon} {self.file}:{self.line} {self.message}"


@dataclass
class Location:
    file: str
    line: int
    column: int = 0


# ── LSP Server Registry ────────────────────────────────────────────

SERVERS: dict[str, dict[str, Any]] = {
    "python": {
        "cmd": ["pyright-langserver", "--stdio"],
        "check": "pyright-langserver",
        "fallback_cmd": ["pylsp"],
        "fallback_check": "pylsp",
        "install": "pip3 install pyright",
    },
    "javascript": {
        "cmd": ["typescript-language-server", "--stdio"],
        "check": "typescript-language-server",
        "install": "npm install -g typescript-language-server typescript",
    },
    "typescript": {
        "cmd": ["typescript-language-server", "--stdio"],
        "check": "typescript-language-server",
        "install": "npm install -g typescript-language-server typescript",
    },
    "go": {
        "cmd": ["gopls", "serve"],
        "check": "gopls",
        "install": "go install golang.org/x/tools/gopls@latest",
    },
    "rust": {
        "cmd": ["rust-analyzer"],
        "check": "rust-analyzer",
        "install": "rustup component add rust-analyzer",
    },
}

EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}


def detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix
    return EXT_TO_LANG.get(ext, "unknown")


def find_available_server(language: str) -> list[str] | None:
    """Find an available LSP server command for the given language."""
    server = SERVERS.get(language)
    if not server:
        return None
    if shutil.which(server["check"]):
        return server["cmd"]
    # Try fallback
    fallback = server.get("fallback_check")
    if fallback and shutil.which(fallback):
        return server.get("fallback_cmd", [])
    return None


# ── LSP Client ──────────────────────────────────────────────────────

class LSPClient:
    """Minimal Language Server Protocol client over stdio.

    Communicates via JSON-RPC 2.0 with Content-Length headers.
    Background thread reads server responses and notifications.
    """

    def __init__(self, language: str, project_root: str) -> None:
        self.language = language
        self.root = os.path.abspath(project_root)
        self.process: subprocess.Popen | None = None
        self._request_id = 0
        self._pending: dict[int, queue.Queue] = {}
        self._diagnostics: dict[str, list[Diagnostic]] = {}  # uri → diagnostics
        self._reader_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start the language server. Returns True on success."""
        cmd = find_available_server(self.language)
        if not cmd:
            return False

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, OSError):
            return False

        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        # LSP initialize handshake
        result = self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": f"file://{self.root}",
            "rootPath": self.root,
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {"relatedInformation": True},
                    "completion": {"completionItem": {"snippetSupport": False}},
                    "hover": {"contentFormat": ["plaintext"]},
                    "definition": {},
                },
                "workspace": {
                    "workspaceFolders": True,
                },
            },
            "workspaceFolders": [
                {"uri": f"file://{self.root}", "name": os.path.basename(self.root)}
            ],
        })

        if result is None:
            self.stop()
            return False

        self._notify("initialized", {})
        return True

    def stop(self) -> None:
        """Shutdown the language server."""
        self._running = False
        if self.process:
            try:
                self._request("shutdown", {})
                self._notify("exit", {})
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        if self._reader_thread:
            self._reader_thread.join(timeout=2)

    @property
    def is_running(self) -> bool:
        return self._running and self.process is not None and self.process.poll() is None

    # ── Public API ──────────────────────────────────────────────────

    def get_diagnostics(self, file_path: str, timeout: float = 3.0) -> list[Diagnostic]:
        """Open a file and wait for diagnostics from the server."""
        uri = self._path_to_uri(file_path)
        try:
            content = Path(file_path).read_text(errors="replace")
        except Exception:
            return []

        # Open the document
        self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": self.language,
                "version": 1,
                "text": content,
            }
        })

        # Wait for diagnostics to arrive
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if uri in self._diagnostics:
                    return list(self._diagnostics[uri])
            time.sleep(0.1)

        return self._diagnostics.get(uri, [])

    def get_definition(self, file_path: str, line: int, col: int) -> Location | None:
        """Go to definition at a position."""
        uri = self._path_to_uri(file_path)
        result = self._request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": col},
        })

        if not result:
            return None

        # Result can be Location, Location[], or LocationLink[]
        if isinstance(result, list) and result:
            loc = result[0]
        elif isinstance(result, dict):
            loc = result
        else:
            return None

        target_uri = loc.get("uri", loc.get("targetUri", ""))
        target_range = loc.get("range", loc.get("targetRange", {}))
        start = target_range.get("start", {})

        return Location(
            file=self._uri_to_path(target_uri),
            line=start.get("line", 0) + 1,
            column=start.get("character", 0),
        )

    def get_hover(self, file_path: str, line: int, col: int) -> str:
        """Get type info / documentation at a position."""
        uri = self._path_to_uri(file_path)
        result = self._request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": col},
        })

        if not result or "contents" not in result:
            return ""

        contents = result["contents"]
        if isinstance(contents, dict):
            return contents.get("value", "")
        if isinstance(contents, str):
            return contents
        if isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, dict):
                    parts.append(item.get("value", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return ""

    def notify_change(self, file_path: str, content: str) -> None:
        """Notify the server that a file changed (call after edit/write)."""
        uri = self._path_to_uri(file_path)
        # Clear old diagnostics
        with self._lock:
            self._diagnostics.pop(uri, None)

        self._notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": int(time.time())},
            "contentChanges": [{"text": content}],
        })

    # ── LSP protocol plumbing ───────────────────────────────────────

    def _request(self, method: str, params: dict, timeout: float = 10.0) -> Any:
        """Send a request and wait for the response."""
        self._request_id += 1
        req_id = self._request_id
        response_queue: queue.Queue = queue.Queue()

        with self._lock:
            self._pending[req_id] = response_queue

        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        self._send(msg)

        try:
            response = response_queue.get(timeout=timeout)
            return response.get("result")
        except queue.Empty:
            with self._lock:
                self._pending.pop(req_id, None)
            return None

    def _notify(self, method: str, params: dict) -> None:
        """Send a notification (no response expected)."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._send(msg)

    def _send(self, msg: dict) -> None:
        """Encode and send a JSON-RPC message."""
        if not self.process or not self.process.stdin:
            return
        body = json.dumps(msg)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        try:
            self.process.stdin.write(header.encode() + body.encode())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self._running = False

    def _read_loop(self) -> None:
        """Background thread: read and route server messages."""
        if not self.process or not self.process.stdout:
            return

        while self._running and self.process.poll() is None:
            try:
                # Read headers
                headers: dict[str, str] = {}
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        self._running = False
                        return
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        break  # end of headers
                    if ":" in line_str:
                        key, value = line_str.split(":", 1)
                        headers[key.strip().lower()] = value.strip()

                # Read body
                length = int(headers.get("content-length", "0"))
                if length == 0:
                    continue
                body = self.process.stdout.read(length)
                if not body:
                    self._running = False
                    return

                msg = json.loads(body.decode("utf-8", errors="replace"))
                self._handle_message(msg)

            except (json.JSONDecodeError, ValueError, KeyError):
                continue
            except Exception:
                if not self._running:
                    return
                continue

    def _handle_message(self, msg: dict) -> None:
        """Route incoming messages to the right handler."""
        if "id" in msg and "method" not in msg:
            # Response to a request we sent
            req_id = msg["id"]
            with self._lock:
                response_queue = self._pending.pop(req_id, None)
            if response_queue:
                response_queue.put(msg)

        elif msg.get("method") == "textDocument/publishDiagnostics":
            # Diagnostics notification from server
            params = msg.get("params", {})
            uri = params.get("uri", "")
            diags = []
            for d in params.get("diagnostics", []):
                severity_map = {1: "error", 2: "warning", 3: "info", 4: "info"}
                start = d.get("range", {}).get("start", {})
                diags.append(Diagnostic(
                    file=self._uri_to_path(uri),
                    line=start.get("line", 0) + 1,
                    severity=severity_map.get(d.get("severity", 3), "info"),
                    message=d.get("message", ""),
                    source=d.get("source", ""),
                ))
            with self._lock:
                self._diagnostics[uri] = diags

        elif msg.get("method") == "window/logMessage":
            pass  # ignore log messages

    @staticmethod
    def _path_to_uri(path: str) -> str:
        return f"file://{os.path.abspath(path)}"

    @staticmethod
    def _uri_to_path(uri: str) -> str:
        if uri.startswith("file://"):
            return uri[7:]
        return uri


# ── Offline Diagnostics (existing functionality) ───────────────────

def get_diagnostics(file_path: Path) -> list[Diagnostic]:
    """Get diagnostics using available offline tools.

    Tries in order: ruff, pyflakes, python -m py_compile
    """
    diagnostics: list[Diagnostic] = []
    rel = file_path.name

    if file_path.suffix != ".py":
        return diagnostics

    # Try ruff (fast Python linter)
    if shutil.which("ruff"):
        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", str(file_path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip():
                for item in json.loads(result.stdout):
                    diagnostics.append(Diagnostic(
                        file=rel,
                        line=item.get("location", {}).get("row", 0),
                        severity="warning",
                        message=f"{item.get('code', '')}: {item.get('message', '')}",
                        source="ruff",
                    ))
            return diagnostics
        except Exception:
            pass

    # Try pyflakes
    if shutil.which("pyflakes"):
        try:
            result = subprocess.run(
                ["pyflakes", str(file_path)],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    try:
                        lineno = int(parts[1])
                    except ValueError:
                        lineno = 0
                    diagnostics.append(Diagnostic(
                        file=rel, line=lineno, severity="warning",
                        message=parts[2].strip(), source="pyflakes",
                    ))
            return diagnostics
        except Exception:
            pass

    # Fallback: py_compile
    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(file_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "syntax error"
            diagnostics.append(Diagnostic(
                file=rel, line=0, severity="error", message=err, source="py_compile",
            ))
    except Exception:
        pass

    return diagnostics


def get_imports_for_symbol(symbol: str, file_path: Path) -> list[str]:
    """Suggest import statements for an undefined symbol."""
    common_imports = {
        "Path": "from pathlib import Path",
        "os": "import os",
        "sys": "import sys",
        "json": "import json",
        "re": "import re",
        "time": "import time",
        "datetime": "import datetime",
        "typing": "from typing import",
        "List": "from typing import List",
        "Dict": "from typing import Dict",
        "Optional": "from typing import Optional",
        "dataclass": "from dataclasses import dataclass",
        "field": "from dataclasses import field",
        "requests": "import requests",
        "numpy": "import numpy as np",
        "np": "import numpy as np",
        "pd": "import pandas as pd",
        "pandas": "import pandas as pd",
        "plt": "import matplotlib.pyplot as plt",
        "torch": "import torch",
        "nn": "from torch import nn",
        "Flask": "from flask import Flask",
        "FastAPI": "from fastapi import FastAPI",
        "pytest": "import pytest",
    }
    if symbol in common_imports:
        return [common_imports[symbol]]
    return []


# ── Enhanced Checker (LSP + offline fallback) ──────────────────────

class EnhancedChecker:
    """Use LSP when available, fall back to offline tools."""

    def __init__(self, project_root: str) -> None:
        self.root = project_root
        self._clients: dict[str, LSPClient] = {}  # language → client
        self._failed_languages: set[str] = set()   # don't retry failed starts

    def check(self, file_path: str) -> dict:
        """Check a file for errors. Returns {"ok": bool, "errors": [...], "warnings": [...]}."""
        lang = detect_language(file_path)
        abs_path = os.path.join(self.root, file_path) if not os.path.isabs(file_path) else file_path

        # Try LSP
        if lang not in self._failed_languages:
            client = self._get_or_start_client(lang)
            if client and client.is_running:
                diagnostics = client.get_diagnostics(abs_path, timeout=5.0)
                errors = [d for d in diagnostics if d.severity == "error"]
                warnings = [d for d in diagnostics if d.severity == "warning"]
                return {
                    "ok": len(errors) == 0,
                    "errors": errors,
                    "warnings": warnings,
                    "error": "\n".join(str(e) for e in errors) if errors else None,
                    "source": "lsp",
                }

        # Fallback: offline tools
        diagnostics = get_diagnostics(Path(abs_path))
        errors = [d for d in diagnostics if d.severity == "error"]
        warnings = [d for d in diagnostics if d.severity == "warning"]
        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "error": "\n".join(str(e) for e in errors) if errors else None,
            "source": "offline",
        }

    def notify_change(self, file_path: str) -> None:
        """Notify LSP server that a file changed."""
        lang = detect_language(file_path)
        client = self._clients.get(lang)
        if client and client.is_running:
            abs_path = os.path.join(self.root, file_path) if not os.path.isabs(file_path) else file_path
            try:
                content = Path(abs_path).read_text(errors="replace")
                client.notify_change(abs_path, content)
            except Exception:
                pass

    def get_definition(self, file_path: str, line: int, col: int) -> Location | None:
        """Go to definition — useful for understanding code."""
        lang = detect_language(file_path)
        client = self._get_or_start_client(lang)
        if client and client.is_running:
            abs_path = os.path.join(self.root, file_path) if not os.path.isabs(file_path) else file_path
            return client.get_definition(abs_path, line, col)
        return None

    def get_hover(self, file_path: str, line: int, col: int) -> str:
        """Get type info at position — useful for context building."""
        lang = detect_language(file_path)
        client = self._get_or_start_client(lang)
        if client and client.is_running:
            abs_path = os.path.join(self.root, file_path) if not os.path.isabs(file_path) else file_path
            return client.get_hover(abs_path, line, col)
        return ""

    def stop_all(self) -> None:
        """Stop all running language servers."""
        for client in self._clients.values():
            try:
                client.stop()
            except Exception:
                pass
        self._clients.clear()

    def _get_or_start_client(self, language: str) -> LSPClient | None:
        if language in self._clients:
            client = self._clients[language]
            if client.is_running:
                return client
            # Dead client — try restarting once
            del self._clients[language]

        if language in self._failed_languages:
            return None

        client = LSPClient(language, self.root)
        if client.start():
            self._clients[language] = client
            return client
        else:
            self._failed_languages.add(language)
            return None
