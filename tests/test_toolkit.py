"""Tests for gem.toolkit — tool registration, schemas, and individual tool execution."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from gem.config import AppConfig, BrowserConfig, LoggingConfig, RuntimeConfig, SafetyConfig, SearchConfig, UIConfig, VoiceConfig
from gem.toolkit import GemTool, GemToolkit


def _make_toolkit(repo_root: Path) -> GemToolkit:
    """Create a GemToolkit with test-appropriate config."""
    config = AppConfig(
        runtime=RuntimeConfig(
            provider="ollama",
            model="test-model",
        ),
        search=SearchConfig(),
        browser=BrowserConfig(),
        voice=VoiceConfig(),
        ui=UIConfig(),
        safety=SafetyConfig(),
        logging=LoggingConfig(),
    )
    return GemToolkit(repo_root, config)


class TestToolRegistration:
    """Verify that GemToolkit registers all expected built-in tools."""

    def test_core_tools_present(self, tmp_repo: Path) -> None:
        """The CORE_TOOLS set should all be registered."""
        tk = _make_toolkit(tmp_repo)
        for name in GemToolkit.CORE_TOOLS:
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
        assert compact_names.issubset(GemToolkit.CORE_TOOLS)

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
    """Verify GemTool.as_schema returns the correct structure."""

    def test_schema_format(self) -> None:
        tool = GemTool(
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
