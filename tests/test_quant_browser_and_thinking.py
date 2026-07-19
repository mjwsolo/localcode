"""Regression tests for the model-quant browser and live reasoning stream.

Covers:
  - hf_quants hides sidecar files (mmproj / mtp / subfolder) so the picker
    never shows tiny "0.5 GB" rows that aren't selectable models
  - ChatLog live thinking stream buffers partial lines and flushes whole ones
"""
from __future__ import annotations


# ── Quant browser: only real weight quants, never sidecars ─────────────

from localcode.hf_quants import _quant_from_entry


def _entry(path: str, size: int):
    return {"path": path, "size": size, "lfs": {"size": size}}


def test_real_weight_quants_are_kept():
    q = _quant_from_entry(_entry("gemma-4-12b-it-UD-Q4_K_XL.gguf", 6_716_356_800))
    assert q is not None
    assert q.label == "UD-Q4_K_XL"
    assert round(q.size_gb, 1) == 6.7


def test_mmproj_sidecar_dropped():
    # The 0.9 GB / 0.5 GB "BF16/F16" rows the user saw were vision projectors.
    for name in ("mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"):
        assert _quant_from_entry(_entry(name, 175_115_840)) is None


def test_mtp_draft_head_dropped():
    assert _quant_from_entry(_entry("mtp-gemma-4-12B-it.gguf", 253_708_800)) is None


def test_subfolder_files_dropped():
    # e.g. an `MTP/…-Q4_0.gguf` draft quant — has a quant code but is a sidecar.
    assert _quant_from_entry(_entry("MTP/gemma-4-12B-Q4_0.gguf", 300_000_000)) is None
    assert _quant_from_entry(_entry("MTP/gemma-4-12B-Q8_0.gguf", 500_000_000)) is None


def test_non_quant_extras_dropped():
    assert _quant_from_entry(_entry("gemma-4-12b-it.gguf", 500_000_000)) is None
    assert _quant_from_entry(_entry("README.md", 1000)) is None


def test_no_catalog_gemma_duplicates():
    # The picker must not show a plain + QAT pair for the same size.
    from localcode.models_catalog import MODEL_GROUPS, CHOICES
    group_names = [g.display_name for g in MODEL_GROUPS]
    assert len(group_names) == len(set(group_names)), f"duplicate group: {group_names}"
    assert not any("qat" in g.key.lower() for g in MODEL_GROUPS)
    assert not any("qat" in c.key.lower() for c in CHOICES)
    # The plain 12B is back and recommendable.
    assert any(c.key == "gemma-12b" for c in CHOICES)


# ── Live reasoning stream: buffer partials, flush whole lines ───────────

class _FakeLog:
    """Minimal stand-in exercising ChatLog's thinking-stream methods."""
    def __init__(self):
        self.written = []
        self._history = []
        self._think_stream_active = False
        self._think_stream_buf = ""

    # Methods copied-by-reference from ChatLog via bind below.
    def write(self, renderable):
        self.written.append(str(getattr(renderable, "plain", renderable)))

    def _track_lines(self):
        pass

    def _dispatch_gap(self, _kind):
        pass

    def _content_width(self):
        return 80


def _bind_stream_methods(obj):
    from localcode.tui.widgets.chat_log import ChatLog
    import types
    for name in ("start_thinking_stream", "stream_thinking",
                 "_commit_thinking_line", "_render_think_header",
                 "_render_think_line", "end_thinking_stream"):
        setattr(obj, name, types.MethodType(getattr(ChatLog, name), obj))


def test_thinking_stream_flushes_whole_lines_only():
    log = _FakeLog()
    _bind_stream_methods(log)
    log.stream_thinking("First reasoning line\nSecond li")
    # Header + first complete line written; partial "Second li" still buffered.
    joined = " ".join(log.written)
    assert "thinking" in joined.lower()
    assert "First reasoning line" in joined
    assert "Second li" not in joined
    assert log._think_stream_buf == "Second li"


def test_thinking_stream_end_flushes_partial():
    log = _FakeLog()
    _bind_stream_methods(log)
    log.stream_thinking("partial tail with no newline")
    assert "partial tail" not in " ".join(log.written)  # buffered, not flushed
    log.end_thinking_stream()
    assert "partial tail with no newline" in " ".join(log.written)
    assert log._think_stream_active is False


def test_thinking_stream_ignores_empty():
    log = _FakeLog()
    _bind_stream_methods(log)
    log.stream_thinking("")
    assert log.written == []


# ── Real-widget smoke test (Textual headless pilot) ────────────────────
#
# Wraps an async scenario in asyncio.run(), matching tests/test_comprehensive_tui
# (no pytest-asyncio plugin needed). Mounts a real ChatLog and drives the live
# reasoning stream to catch rendering exceptions the fake-log tests can't.

import asyncio


def test_thinking_stream_renders_in_real_chatlog():
    from textual.app import App, ComposeResult
    from localcode.tui.widgets.chat_log import ChatLog

    class _App(App):
        def compose(self) -> ComposeResult:
            yield ChatLog(id="chat-log")

    async def scenario():
        app = _App()
        async with app.run_test() as pilot:
            await pilot.pause()
            log = app.query_one("#chat-log", ChatLog)
            log.stream_thinking("Analyze the request.\nPlan the file layout.\nStill wr")
            await pilot.pause()
            assert log._think_stream_active is True
            assert log._think_stream_buf == "Still wr"
            log.end_thinking_stream()
            await pilot.pause()
            assert log._think_stream_active is False

    asyncio.run(scenario())
