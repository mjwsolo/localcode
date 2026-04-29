from localcode.tui.screens.chat import _NoTintInput
from localcode.tui.widgets.chat_log import ChatLog


class _Selection:
    is_empty = True


class _PasteEvent:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prevented = False
        self.stopped = False

    def prevent_default(self) -> None:
        self.prevented = True

    def stop(self) -> None:
        self.stopped = True


class _InputLike:
    selection = _Selection()

    def __init__(self) -> None:
        self.inserted = ""

    def insert_text_at_cursor(self, text: str) -> None:
        self.inserted += text


def test_chat_input_paste_prevents_textual_default_insert() -> None:
    event = _PasteEvent("hello")
    target = _InputLike()

    _NoTintInput._on_paste(target, event)

    assert event.prevented
    assert event.stopped
    assert target.inserted == "hello"


def test_chat_input_paste_preserves_multiline_content_as_single_line() -> None:
    event = _PasteEvent("line one\nline two\r\nline three")
    target = _InputLike()

    _NoTintInput._on_paste(target, event)

    assert target.inserted == "line one line two line three"


class _HistInputLike:
    """Minimal stub for _hist_navigate without spinning up Textual."""

    def __init__(self, value: str = "") -> None:
        self.value = value
        self.cursor_position = len(value)
        self._input_history: list[str] = []
        self._input_history_pos: int = -1
        self._input_history_draft: str = ""

    def _hist_init(self) -> None:
        pass


def test_down_arrow_clears_draft_when_not_browsing_history() -> None:
    target = _HistInputLike(value="some half-typed prompt")
    changed = _NoTintInput._hist_navigate(target, +1)
    assert changed is True
    assert target.value == ""
    assert target.cursor_position == 0
    assert target._input_history_pos == -1


def test_down_arrow_on_empty_input_lets_key_bubble() -> None:
    target = _HistInputLike(value="")
    changed = _NoTintInput._hist_navigate(target, +1)
    assert changed is False
    assert target.value == ""


def test_down_arrow_while_browsing_history_walks_forward_not_clear() -> None:
    target = _HistInputLike(value="older entry")
    target._input_history = ["older entry", "newer entry"]
    target._input_history_pos = 0
    changed = _NoTintInput._hist_navigate(target, +1)
    assert changed is True
    assert target.value == "newer entry"
    assert target._input_history_pos == 1


def _thinking_log() -> ChatLog:
    log = ChatLog()
    log._history = [("thinking", "first line\nsecond line")]
    log._thinking_states = {0: False}
    log._thinking_line_map = {3: 0}
    log._rerender = lambda: None  # type: ignore[method-assign]
    return log


def test_thinking_disclosure_click_toggles_matching_row() -> None:
    log = _thinking_log()

    handled = log._handle_thinking_click(3, x=3)

    assert handled is True
    assert log._thinking_states[0] is True


def test_thinking_disclosure_duplicate_click_only_toggles_once() -> None:
    log = _thinking_log()

    assert log._handle_thinking_click(3, x=3) is True
    assert log._handle_thinking_click(3, x=3) is True

    assert log._thinking_states[0] is True


def test_thinking_gutter_click_far_from_header_does_not_toggle() -> None:
    log = _thinking_log()

    handled = log._handle_thinking_click(40, x=3)

    assert handled is False
    assert log._thinking_states[0] is False
