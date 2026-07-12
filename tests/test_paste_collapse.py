"""Unit tests for the pure paste-collapse helpers.

These import ONLY `localcode.tui.paste_collapse`, which has no Textual
dependency, so they run headlessly (the dev env has no `textual`).
"""
from localcode.tui.paste_collapse import (
    LARGE_PASTE_MIN_CHARS,
    LARGE_PASTE_MIN_LINES,
    PasteBuffer,
    is_large_paste,
)


# ── is_large_paste thresholds ──

def test_small_paste_stays_inline():
    assert is_large_paste("") is False
    assert is_large_paste("hello world") is False
    # 3 lines is below the 4-line floor.
    assert is_large_paste("a\nb\nc") is False
    # 299 chars is below the 300-char floor.
    assert is_large_paste("x" * (LARGE_PASTE_MIN_CHARS - 1)) is False


def test_large_paste_by_lines():
    assert LARGE_PASTE_MIN_LINES == 4
    assert is_large_paste("a\nb\nc\nd") is True  # exactly 4 lines
    assert is_large_paste("a\n" * 50) is True


def test_large_paste_by_chars():
    assert is_large_paste("x" * LARGE_PASTE_MIN_CHARS) is True
    assert is_large_paste("x" * 5000) is True


def test_trailing_newline_does_not_inflate_line_count():
    # "a\nb\nc\n" is 3 lines with a trailing newline, not 4 → still small.
    assert is_large_paste("a\nb\nc\n") is False


# ── PasteBuffer.add / expand ──

def test_chip_shows_line_count():
    pb = PasteBuffer()
    chip = pb.add("l1\nl2\nl3\nl4\nl5")
    assert chip == "[pasted #1 +5 lines]"


def test_chip_shows_char_count_for_single_line():
    pb = PasteBuffer()
    blob = "x" * 400
    chip = pb.add(blob)
    assert chip == "[pasted #1 +400 chars]"


def test_expand_restores_real_text():
    pb = PasteBuffer()
    real = "line1\nline2\nline3\nline4"
    chip = pb.add(real)
    composed = f"here is my code {chip} please review"
    assert pb.expand(composed) == f"here is my code {real} please review"


def test_multiple_pastes_get_unique_chips_and_expand_independently():
    pb = PasteBuffer()
    a = "A\n" * 10
    b = "B\n" * 20
    chip_a = pb.add(a)
    chip_b = pb.add(b)
    assert chip_a == "[pasted #1 +10 lines]"
    assert chip_b == "[pasted #2 +20 lines]"
    composed = f"{chip_a} and also {chip_b}"
    expanded = pb.expand(composed)
    assert expanded == f"{a} and also {b}"


def test_deleted_chip_text_is_dropped():
    pb = PasteBuffer()
    a = "keep\nme\nplease\nnow"
    b = "drop\nme\nplease\nnow"
    chip_a = pb.add(a)
    chip_b = pb.add(b)
    # User deleted chip_b from the composer before submitting.
    composed = f"only {chip_a} survives"
    expanded = pb.expand(composed)
    assert a in expanded
    assert b not in expanded
    assert chip_b not in expanded


def test_prune_forgets_deleted_chips():
    pb = PasteBuffer()
    chip_a = pb.add("a\nb\nc\nd")
    chip_b = pb.add("e\nf\ng\nh")
    assert len(pb) == 2
    pb.prune(f"just {chip_a}")
    assert len(pb) == 1
    # The surviving chip still expands.
    assert pb.expand(chip_a) == "a\nb\nc\nd"
    assert pb.expand(chip_b) == chip_b  # no longer known → left as-is


def test_clear_forgets_everything_and_resets_counter():
    pb = PasteBuffer()
    pb.add("a\nb\nc\nd")
    pb.clear()
    assert len(pb) == 0
    # Counter resets so the next message starts at #1 again.
    assert pb.add("e\nf\ng\nh") == "[pasted #1 +4 lines]"


def test_expand_noop_when_empty():
    pb = PasteBuffer()
    assert pb.expand("nothing to expand") == "nothing to expand"
    assert pb.expand("") == ""
