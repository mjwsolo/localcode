"""Streaming must render append-only WITHOUT changing what the user sees.

`_flush_stream` no longer re-renders the whole message each frame; it freezes
everything up to the last safe block boundary and re-renders only the tail.
That is only sound if rendering the pieces separately produces exactly the same
rows as rendering the whole text at once - otherwise the "snap to even spacing"
bug at end-of-turn comes straight back, because history replay renders whole.

These tests pin that equivalence, and pin the boundary rules that make it hold.
"""

from io import StringIO

import pytest

from localcode.tui.widgets.chat_log import ChatLog

stable_split = ChatLog._stable_split


def render(text: str, width: int = 100) -> list[str]:
    """Mirror of ChatLog._render_assistant_to_lines' markdown branch."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text

    buf = StringIO()
    console = Console(
        file=buf, width=width - 2, force_terminal=True, color_system="truecolor"
    )
    console.print(Markdown(text, code_theme="monokai"))
    out = buf.getvalue()
    if out.endswith("\n"):
        out = out[:-1]
    rows = [Text.from_ansi(f"  {line}").plain for line in out.split("\n")]
    while rows and not rows[-1].strip():
        rows.pop()
    return rows


def render_piecewise(text: str, width: int = 100) -> list[str]:
    """What the streaming path actually puts on screen."""
    cut = stable_split(text)
    if not cut:
        return render(text, width)
    rows = render(text[:cut], width)
    tail = text[cut:]
    if tail.strip():
        rows += render(tail, width)
    return rows


PROSE = "Here is a paragraph explaining the change in some detail.\n\n"
FENCE = "```python\ndef f(x):\n    return x * 2\n```\n\n"
HEADING = "### A heading\n\n"
LIST = "- point one\n- point two\n\n"


@pytest.mark.parametrize(
    "text",
    [
        PROSE * 3,
        PROSE + FENCE + PROSE,
        HEADING + PROSE + FENCE,
        PROSE + FENCE + HEADING + PROSE + FENCE,
        (PROSE + FENCE) * 4,
    ],
    ids=["prose", "prose-fence", "heading", "mixed", "long"],
)
def test_piecewise_matches_whole(text):
    assert render_piecewise(text) == render(text)


def test_never_splits_inside_a_code_fence():
    # A blank line inside an open fence is not a block boundary.
    text = PROSE + "```python\ndef f():\n\n    return 1\n```\n\n" + PROSE
    cut = stable_split(text)
    assert text[:cut].count("```") % 2 == 0
    assert render_piecewise(text) == render(text)


def test_never_splits_a_list():
    # Splitting between list items would make CommonMark render the list
    # "loose" and add spacing the whole-render never produces.
    text = PROSE + LIST + LIST + PROSE
    assert render_piecewise(text) == render(text)


def test_no_stable_point_falls_back_to_whole():
    assert stable_split("one line, still streaming") == 0
    assert stable_split("") == 0


def test_split_point_only_advances():
    # The streaming loop relies on the boundary being monotonic as text grows.
    text = ""
    last = 0
    for piece in (PROSE, FENCE, HEADING, PROSE, LIST, PROSE):
        text += piece
        cut = stable_split(text)
        assert cut >= last
        last = cut
