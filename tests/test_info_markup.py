"""append_info must render Rich markup, not print the tags literally.

The PTY QA pass caught info lines showing raw `[dim]…[/]` tags to the user -
most visibly the approval hint (`[dim]press 1/y allow…[/]`) shown on every
approval prompt, and the resumed-session header. _render_info wrote the string
as a literal Text, so the tags never styled anything.

The fix parses markup. That makes escaping matter: the resume view interpolates
the user's / model's own words (which can contain `arr[0]`, `[TODO]`) into a
markup string, so those callers escape the content. And malformed markup must
never crash the render.
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult

from localcode.tui.widgets.chat_log import ChatLog


class _Host(App):
    def compose(self) -> ComposeResult:
        yield ChatLog(id="chat-log")


def _render(*lines: str) -> str:
    """Append each info line and return the widget's rendered plain text."""
    out = {}

    async def scenario():
        app = _Host()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            log = app.query_one("#chat-log", ChatLog)
            for ln in lines:
                log.append_info(ln)
            await pilot.pause(0.05)
            out["text"] = "\n".join(
                "".join(seg.text for seg in line._segments)
                if hasattr(line, "_segments")
                else str(line)
                for line in log.lines
            )

    asyncio.run(scenario())
    return out["text"]


def test_markup_tags_are_not_shown_literally():
    rendered = _render("[dim]press 1/y allow · 2 always · 3/n/Esc deny · 4 stop asking[/]")
    assert "[dim]" not in rendered
    assert "[/]" not in rendered
    # The actual words survive.
    assert "press 1/y allow" in rendered
    assert "stop asking" in rendered


def test_resume_header_markup_renders():
    rendered = _render("[dim]── resumed session (3 messages) ──[/]")
    assert "[dim]" not in rendered and "[/]" not in rendered
    assert "resumed session (3 messages)" in rendered


def test_escaped_user_content_keeps_literal_brackets():
    # Mirrors the resume-view caller: markup wrapper + escaped user text.
    from rich.markup import escape

    rendered = _render(f"[dim]you:[/] {escape('arr[0] = [TODO] fix it')}")
    assert "[dim]" not in rendered and "[/]" not in rendered
    assert "you:" in rendered
    # The user's brackets are shown verbatim, not swallowed as tags.
    assert "arr[0] = [TODO] fix it" in rendered


def test_malformed_markup_does_not_crash():
    # An unbalanced bracket must fall back to literal, never raise.
    rendered = _render("weird [not a real tag value")
    assert "weird" in rendered
    assert "not a real tag value" in rendered
