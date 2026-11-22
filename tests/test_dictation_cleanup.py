"""Dictation cleanup removes whisper's non-speech labels and leaves code alone."""
import pytest

from localcode.ui.supervisor import strip_non_speech_captions as clean


@pytest.mark.parametrize("raw,expected", [
    ("Use items[0] and call reset().", "Use items[0] and call reset()."),
    ("call foo(x) then bar[1]", "call foo(x) then bar[1]"),
    ("[BLANK_AUDIO]", ""),
    ("hello [music] world", "hello world"),
    ("(laughs) okay (Wind Blowing)", "okay"),
    ("see (the docs) please", "see (the docs) please"),
])
def test_clean(raw, expected):
    assert clean(raw) == expected
