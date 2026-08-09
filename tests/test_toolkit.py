"""Tests for localcode.toolkit — tool registration, schemas, and individual tool execution."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from localcode.config import AppConfig, LoggingConfig, RuntimeConfig, SafetyConfig, SearchConfig, UIConfig
from localcode.toolkit import LocalCodeTool, LocalCodeToolkit


def _make_toolkit(repo_root: Path) -> LocalCodeToolkit:
    """Create a LocalCodeToolkit with test-appropriate config."""
    config = AppConfig(
        runtime=RuntimeConfig(
            provider="ollama",
            model="test-model",
        ),
        search=SearchConfig(),
        ui=UIConfig(),
        safety=SafetyConfig(),
        logging=LoggingConfig(),
    )
    return LocalCodeToolkit(repo_root, config)


class TestToolRegistration:
    """Verify that LocalCodeToolkit registers all expected built-in tools."""

    def test_core_tools_present(self, tmp_repo: Path) -> None:
        """The CORE_TOOLS set should all be registered."""
        tk = _make_toolkit(tmp_repo)
        for name in LocalCodeToolkit.CORE_TOOLS:
            assert name in tk.tools, f"Core tool '{name}' not registered"

    def test_all_expected_tools_registered(self, tmp_repo: Path) -> None:
        """Key tools that the agent loop depends on should exist."""
        tk = _make_toolkit(tmp_repo)
        expected = [
            "read_file", "write_file", "edit_file", "multi_edit",
            "grep", "glob", "list_files", "bash",
            "current_datetime", "git_status", "git_diff",
            "web_search", "codemod", "run_tests", "repl",
        ]
        for name in expected:
            assert name in tk.tools, f"Tool '{name}' not registered"

    def test_list_tool_names_sorted(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        names = tk.list_tool_names()
        assert names == sorted(names)
        assert len(names) > 10


class TestSchemas:
    """Verify schema generation for tools."""

    def test_schemas_return_list(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        schemas = tk.schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0

    def test_schema_structure(self, tmp_repo: Path) -> None:
        """Each schema should have type=function and function.name/description/parameters."""
        tk = _make_toolkit(tmp_repo)
        for schema in tk.schemas():
            assert schema["type"] == "function"
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_compact_schemas_are_subset(self, tmp_repo: Path) -> None:
        """compact=True should only return CORE_TOOLS."""
        tk = _make_toolkit(tmp_repo)
        full = tk.schemas(compact=False)
        compact = tk.schemas(compact=True)
        assert len(compact) < len(full)
        compact_names = {s["function"]["name"] for s in compact}
        assert compact_names.issubset(LocalCodeToolkit.CORE_TOOLS)

    def test_minimal_schemas_shorter_descriptions(self, tmp_repo: Path) -> None:
        """minimal=True should produce shorter descriptions (first sentence only)."""
        tk = _make_toolkit(tmp_repo)
        full = tk.schemas(compact=False, minimal=False)
        minimal = tk.schemas(compact=False, minimal=True)
        # Find a tool with a multi-sentence description
        for f_schema, m_schema in zip(full, minimal):
            if f_schema["function"]["name"] == m_schema["function"]["name"]:
                # Minimal desc should be <= full desc length
                assert len(m_schema["function"]["description"]) <= len(f_schema["function"]["description"]) + 1


class TestGemToolAsSchema:
    """Verify LocalCodeTool.as_schema returns the correct structure."""

    def test_schema_format(self) -> None:
        tool = LocalCodeTool(
            name="test_tool",
            description="A test tool for testing.",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            handler=lambda args: "ok",
        )
        schema = tool.as_schema()
        assert schema == {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool for testing.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        }


class TestBashTool:
    """Verify the bash tool executes commands and returns output."""

    def test_echo_command(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["bash"].handler({"command": "echo hello_world"})
        assert "hello_world" in result

    def test_failing_command(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["bash"].handler({"command": "exit 1"})
        # Should contain exit code info, not raise
        assert result is not None

    def test_timeout_parameter(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        # Quick command with explicit timeout
        result = tk.tools["bash"].handler({"command": "echo fast", "timeout": 5})
        assert "fast" in result


class TestReadFileTool:
    """Verify the read_file tool returns file content."""

    def test_reads_existing_file(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["read_file"].handler({"path": "main.py"})
        assert "def hello" in result

    def test_missing_file_returns_error(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["read_file"].handler({"path": "nonexistent.py"})
        # Should return an error message, not crash
        assert "error" in result.lower() or "not found" in result.lower() or "Error" in result


class TestGrepTool:
    """Verify the grep tool searches file contents."""

    def test_finds_pattern(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["grep"].handler({"pattern": "def hello"})
        assert "main.py" in result
        assert "def hello" in result

    def test_no_matches(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["grep"].handler({"pattern": "ZZZZNONEXISTENT"})
        assert "no match" in result.lower() or result.strip() == "" or "0" in result


class TestGlobTool:
    """Verify the glob tool finds files by pattern."""

    def test_finds_python_files(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["glob"].handler({"pattern": "**/*.py"})
        assert "main.py" in result
        assert "utils.py" in result

    def test_no_matches(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["glob"].handler({"pattern": "**/*.rs"})
        assert "no match" in result.lower() or result.strip() == "" or "No" in result


class TestCurrentDatetimeTool:
    """Verify the current_datetime tool returns a timestamp."""

    def test_returns_datetime_string(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["current_datetime"].handler({})
        # Should contain date-like patterns
        assert "202" in result  # year prefix


class TestWriteFileTool:
    """Verify the write_file tool creates/overwrites files."""

    def test_creates_new_file(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        result = tk.tools["write_file"].handler({
            "path": "new_file.py",
            "content": "print('created')\n",
        })
        assert (tmp_repo / "new_file.py").exists()
        assert (tmp_repo / "new_file.py").read_text() == "print('created')\n"

    def test_overwrites_existing_file(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        tk.tools["write_file"].handler({
            "path": "main.py",
            "content": "# replaced\n",
        })
        assert (tmp_repo / "main.py").read_text() == "# replaced\n"


class TestEditFileTool:
    """Verify the edit_file tool does surgical string replacement."""

    def test_replaces_exact_match(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        # Need to read first so the tool tracks state
        tk.tools["read_file"].handler({"path": "main.py"})
        result = tk.tools["edit_file"].handler({
            "path": "main.py",
            "old_string": "return 'world'",
            "new_string": "return 'earth'",
        })
        content = (tmp_repo / "main.py").read_text()
        assert "return 'earth'" in content
        assert "return 'world'" not in content


class TestExecuteToolCalls:
    """Verify execute_tool_calls dispatches correctly."""

    def test_executes_single_tool(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        results = tk.execute_tool_calls([
            {
                "function": {
                    "name": "current_datetime",
                    "arguments": {},
                },
            }
        ])
        assert len(results) == 1
        assert "202" in results[0]["content"]

    def test_unknown_tool_returns_error(self, tmp_repo: Path) -> None:
        tk = _make_toolkit(tmp_repo)
        results = tk.execute_tool_calls([
            {
                "function": {
                    "name": "nonexistent_tool",
                    "arguments": {},
                },
            }
        ])
        assert len(results) == 1
        assert "error" in results[0]["content"].lower() or "unknown" in results[0]["content"].lower() or "not" in results[0]["content"].lower()


# A REAL MCP server (official SDK) used by the MCP wiring tests below.
# Built with mcp.server.fastmcp.FastMCP, exposing one `echo` tool, run over
# stdio. The toolkit connects to it through the SDK client (localcode.mcp).
_FAKE_MCP_SERVER = '''#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the arguments back."""
    import json
    return "ECHO:" + json.dumps({"text": text})


if __name__ == "__main__":
    mcp.run()
'''

# OAuth / HTTP / SSE transports are hard to exercise live in CI (they need a
# running remote server and, for OAuth, an interactive browser). The tests
# below therefore only validate that those config paths construct a client
# of the right transport — they are explicitly NOT live-tested end to end.
_HTTP_CONFIG = {"transport": "http", "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer x"}, "oauth": True}
_SSE_CONFIG = {"transport": "sse", "url": "https://example.com/sse", "headers": {}}


class TestMCPWiring:
    """Verify MCP servers connect and their tools become callable."""

    def _setup_fake_mcp(self, tmp_path: Path, monkeypatch) -> str:
        import stat
        from localcode import mcp as mcp_mod

        # Reset the process-wide client registry so prior tests don't leak.
        mcp_mod.shutdown_all()

        home = tmp_path / "lc_home"
        home.mkdir()
        server_py = home / "fake_server.py"
        server_py.write_text(_FAKE_MCP_SERVER)
        server_py.chmod(server_py.stat().st_mode | stat.S_IEXEC)

        config = {
            "mcpServers": {
                "fake": {
                    "command": sys.executable,
                    "args": [str(server_py)],
                    "env": {},
                }
            }
        }
        (home / "mcp.json").write_text(__import__("json").dumps(config))
        monkeypatch.setenv("LOCALCODE_HOME", str(home))
        return "mcp_fake_echo"

    def test_mcp_tool_registered_and_callable(self, tmp_repo: Path, tmp_path: Path, monkeypatch) -> None:
        from localcode import mcp as mcp_mod
        tool_name = self._setup_fake_mcp(tmp_path, monkeypatch)
        try:
            tk = _make_toolkit(tmp_repo)

            # Appears in schemas()
            names_in_schema = {s["function"]["name"] for s in tk.schemas()}
            assert tool_name in names_in_schema, names_in_schema

            # Appears in list_tool_names()
            assert tool_name in tk.list_tool_names()

            # Schema carries the MCP tool's inputSchema
            schema = next(s for s in tk.schemas() if s["function"]["name"] == tool_name)
            assert "text" in schema["function"]["parameters"]["properties"]

            # execute_tool_calls actually invokes the server and echoes args
            results = tk.execute_tool_calls([
                {"function": {"name": tool_name, "arguments": {"text": "hi"}}}
            ])
            assert len(results) == 1
            assert "ECHO:" in results[0]["content"]
            assert "hi" in results[0]["content"]
        finally:
            mcp_mod.shutdown_all()

    def test_ensure_mcp_tools_runs_once(self, tmp_repo: Path, tmp_path: Path, monkeypatch) -> None:
        from localcode import mcp as mcp_mod
        self._setup_fake_mcp(tmp_path, monkeypatch)
        try:
            tk = _make_toolkit(tmp_repo)
            assert tk._mcp_loaded is False
            tk.ensure_mcp_tools()
            assert tk._mcp_loaded is True
            n_after_first = len(tk.tools)
            tk.ensure_mcp_tools()  # idempotent — no duplicate registration
            assert len(tk.tools) == n_after_first
            assert "fake" in tk.mcp_clients
        finally:
            mcp_mod.shutdown_all()


class TestMCPTransportConfig:
    """HTTP / SSE / OAuth config paths.

    These transports need a live remote server (and OAuth an interactive
    browser), so they are NOT live-tested. We only assert the config parses
    into a client of the right transport and that the SDK transport context
    can be constructed without raising.
    """

    def test_http_config_builds_http_client(self) -> None:
        from localcode import mcp as mcp_mod
        cli = mcp_mod._client_from_config("h", _HTTP_CONFIG)
        assert cli.transport == "http"
        assert cli.url == "https://example.com/mcp"
        assert cli.headers["Authorization"] == "Bearer x"
        assert cli.oauth is True
        # Constructing the transport context manager must not raise (it does
        # not connect until entered). This also exercises the OAuth provider
        # construction path.
        cm = cli._make_transport()
        assert cm is not None

    def test_sse_config_builds_sse_client(self) -> None:
        from localcode import mcp as mcp_mod
        cli = mcp_mod._client_from_config("s", _SSE_CONFIG)
        assert cli.transport == "sse"
        assert cli.url == "https://example.com/sse"
        assert cli.oauth is False
        cm = cli._make_transport()
        assert cm is not None

    def test_default_transport_is_stdio(self) -> None:
        from localcode import mcp as mcp_mod
        cli = mcp_mod._client_from_config("d", {"command": "echo", "args": []})
        assert cli.transport == "stdio"

    def test_unknown_transport_raises_on_build(self) -> None:
        from localcode import mcp as mcp_mod
        cli = mcp_mod._client_from_config("u", {"transport": "carrier-pigeon",
                                                "url": "https://x"})
        with pytest.raises(RuntimeError):
            cli._make_transport()


def test_glob_excludes_noise_before_truncating(tmp_path):
    """Regression: the noise-dir exclusion must run BEFORE the 100-item cut,
    or a repo whose newest 100 matches are node_modules/.venv hides real source."""
    import types
    from localcode.tools import glob_tool
    (tmp_path / "real_module.py").write_text("x = 1\n")          # older mtime
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    for i in range(120):                                          # newer, would fill first 100
        (nm / f"f{i}.py").write_text("y\n")
    out = glob_tool.execute(types.SimpleNamespace(repo=tmp_path), {"pattern": "**/*.py"})
    assert "real_module.py" in out          # was hidden past the [:100] cut
    assert "node_modules" not in out        # noise excluded
