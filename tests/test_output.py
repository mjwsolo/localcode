"""Tests for localcode.output — OutputManager phase transitions and formatting."""
from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest

from localcode.output import TOOL_HEADERS, OutputManager, OutputState, Phase, ToolAction


class TestPhaseTransitions:
    """Verify the OutputManager moves through phases correctly."""

    def test_initial_phase_is_idle(self) -> None:
        om = OutputManager()
        assert om.state.phase == Phase.IDLE

    def test_start_thinking_transitions_to_thinking(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            om.start_thinking()
        assert om.state.phase == Phase.THINKING
        om._stop_indicator()

    def test_start_streaming_transitions_to_streaming(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            om.start_thinking()
            om.start_streaming()
        assert om.state.phase == Phase.STREAMING
        om._stop_indicator()

    def test_done_transitions_to_done(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            om.start_thinking()
            om.start_streaming()
            om.done()
        assert om.state.phase == Phase.DONE

    def test_set_error_transitions_to_error(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            om.start_thinking()
            om.set_error("something broke")
        assert om.state.phase == Phase.ERROR
        assert om.state.error == "something broke"

    def test_thinking_reset_clears_state(self) -> None:
        om = OutputManager()
        om.state.tokens = 100
        om.state.content_chunks = ["old"]
        with patch("sys.stdout", new_callable=io.StringIO):
            om.start_thinking(reset=True)
        assert om.state.tokens == 0
        assert om.state.content_chunks == []
        om._stop_indicator()

    def test_thinking_no_reset_preserves_tokens(self) -> None:
        om = OutputManager()
        om.state.tokens = 100
        om.state.start_time = 1000.0
        with patch("sys.stdout", new_callable=io.StringIO):
            om.start_thinking(reset=False)
        assert om.state.tokens == 100
        om._stop_indicator()


class TestLogTool:
    """Verify log_tool records tool actions and outputs formatted headers."""

    def test_log_tool_appends_action(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            idx = om.log_tool("bash", "ls -la")
        assert idx == 0
        assert len(om.state.tool_actions) == 1
        assert om.state.tool_actions[0].name == "bash"
        assert om.state.tool_actions[0].args == "ls -la"
        om._stop_indicator()

    def test_log_tool_formats_header(self) -> None:
        om = OutputManager()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            om.log_tool("bash", "echo hello")
        output = buf.getvalue()
        # Should contain the "Ran" header for bash
        assert "Ran" in output
        om._stop_indicator()

    def test_log_tool_uses_tool_headers(self) -> None:
        """Each registered tool name maps to a human-readable header."""
        assert TOOL_HEADERS["bash"] == "Ran"
        assert TOOL_HEADERS["read_file"] == "Read"
        assert TOOL_HEADERS["write_file"] == "Wrote"
        assert TOOL_HEADERS["edit_file"] == "Edited"
        assert TOOL_HEADERS["grep"] == "Searched"

    def test_multiple_tools_increment_index(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            i0 = om.log_tool("read_file", "main.py")
            i1 = om.log_tool("bash", "pytest")
        assert i0 == 0
        assert i1 == 1
        om._stop_indicator()


class TestToolResult:
    """Verify tool_result records results and outputs tree-indented lines."""

    def test_tool_result_updates_action(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            idx = om.log_tool("bash", "ls")
            om.tool_result("file1.py\nfile2.py", idx=idx)
        assert om.state.tool_actions[0].status == "done"
        assert "file1.py" in om.state.tool_actions[0].result
        om._stop_indicator()

    def test_tool_result_error_flag(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            idx = om.log_tool("bash", "false")
            om.tool_result("command failed", error=True, idx=idx)
        assert om.state.tool_actions[0].status == "error"
        om._stop_indicator()

    def test_tool_result_tree_indentation(self) -> None:
        """Output should use tree connector characters."""
        om = OutputManager()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            idx = om.log_tool("bash", "echo hi")
            om.tool_result("hello world", idx=idx)
        output = buf.getvalue()
        assert "\u2514" in output or "└" in output  # tree connector
        om._stop_indicator()

    def test_long_result_truncated(self) -> None:
        """Results with more than 8 lines should show a '... +N lines' footer."""
        om = OutputManager()
        buf = io.StringIO()
        lines = "\n".join(f"line_{i}" for i in range(20))
        with patch("sys.stdout", buf):
            idx = om.log_tool("bash", "big output")
            om.tool_result(lines, idx=idx)
        output = buf.getvalue()
        assert "+12 lines" in output or "+12" in output
        om._stop_indicator()


class TestSetError:
    """Verify set_error formats the error message with red coloring."""

    def test_error_message_appears(self) -> None:
        om = OutputManager()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            om.set_error("timeout after 120s")
        output = buf.getvalue()
        assert "timeout after 120s" in output
        assert "Error" in output


class TestStream:
    """Verify stream() accumulates content chunks."""

    def test_stream_accumulates_chunks(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            om.start_thinking()
            om.start_streaming()
            om.stream("hello ")
            om.stream("world")
        assert om.state.content_chunks == ["hello ", "world"]

    def test_stream_auto_transitions_to_streaming(self) -> None:
        om = OutputManager()
        with patch("sys.stdout", new_callable=io.StringIO):
            om.start_thinking()
            # stream() should auto-transition if not already STREAMING
            om.stream("data")
        assert om.state.phase == Phase.STREAMING


class TestFeedThinking:
    """Verify feed_thinking increments approximate token count."""

    def test_increments_tokens(self) -> None:
        om = OutputManager()
        om.state.tokens = 0
        om.feed_thinking("a" * 40)
        assert om.state.tokens >= 1


class TestFilterNoise:
    """Verify _filter_noise strips macOS MallocStackLogging warnings."""

    def test_strips_malloc_noise(self) -> None:
        text = "real output\nMallocStackLogging: can't turn off\nmore output"
        result = OutputManager._filter_noise(text)
        assert "MallocStackLogging" not in result
        assert "real output" in result
        assert "more output" in result
