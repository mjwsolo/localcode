"""The todo_write tool line must render clean, not dump its raw args dict.

The QA pass caught `todo_write` rendering its whole `{'todos': [...]}` argument
blob on the tool line (it had no entry in _TOOL_HEADERS and no arg suppression),
overflowing the row. It now renders a friendly `Todo` name with the args hidden;
the result summary carries the progress.
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult

from localcode.tui.widgets.chat_log import ChatLog


class _Host(App):
    def compose(self) -> ComposeResult:
        yield ChatLog(id="chat-log")


def _render_todo(summary: str) -> str:
    out = {}
    raw_args = (
        "{'todos': [{'content': 'Create hello.py with a function that greets', "
        "'status': 'in_progress'}]}"
    )

    async def scenario():
        app = _Host()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            log = app.query_one("#chat-log", ChatLog)
            log._render_tool("todo_write", raw_args)
            log._render_tool_done("todo_write", raw_args, summary)
            await pilot.pause(0.05)
            out["text"] = "\n".join(
                "".join(seg.text for seg in line._segments)
                if hasattr(line, "_segments") else str(line)
                for line in log.lines
            )

    asyncio.run(scenario())
    return out["text"]


def test_todo_line_hides_raw_dict_and_uses_friendly_name():
    rendered = _render_todo("Task list updated: 0/1 done, 1 in progress.")
    assert "{" not in rendered and "'todos'" not in rendered
    assert "Todo" in rendered
    assert "0/1 done" in rendered


def test_todo_summary_string_has_no_em_dash():
    # The summary localcode itself builds (todo_write.run) must use a colon,
    # not an em-dash, on the user-facing line.
    import inspect

    from localcode.tools import todo_write

    src = inspect.getsource(todo_write)
    assert "Task list updated —" not in src
    assert "Task list updated:" in src
