"""Auto-compaction fires relative to the machine's real context window, and is
user-tunable (like Claude Code's CLAUDE_AUTOCOMPACT_PCT_OVERRIDE)."""
import os
import pytest

from localcode.compaction import _compact_fraction, should_compact


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("LOCALCODE_COMPACT_PCT", raising=False)


def _msgs(tokens):
    return [{"role": "user", "content": "x" * (4 * tokens)}]


def test_default_fraction_is_070():
    assert _compact_fraction() == 0.70


def test_threshold_scales_with_window():
    m = _msgs(90_000)  # ~90k tokens
    # 128k window: 90k is above ~89k threshold → compact.
    assert should_compact(m, 131072) is True
    # 256k window: 90k is well below ~176k threshold → don't compact.
    assert should_compact(m, 262144) is False


def test_env_override_percent(monkeypatch):
    monkeypatch.setenv("LOCALCODE_COMPACT_PCT", "90")
    assert _compact_fraction() == 0.90
    # At 90%, 90k tokens on a 128k window no longer compacts (~114k threshold).
    assert should_compact(_msgs(90_000), 131072) is False


def test_env_override_fraction_form(monkeypatch):
    monkeypatch.setenv("LOCALCODE_COMPACT_PCT", "0.5")
    assert _compact_fraction() == 0.5


def test_env_override_clamped(monkeypatch):
    monkeypatch.setenv("LOCALCODE_COMPACT_PCT", "999")
    assert _compact_fraction() == 0.95  # clamped
    monkeypatch.setenv("LOCALCODE_COMPACT_PCT", "junk")
    assert _compact_fraction() == 0.70  # malformed → default
