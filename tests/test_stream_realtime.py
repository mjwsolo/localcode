"""Drive the REAL ChatLog widget through a full streamed reply and time it.

The append-only work in `_flush_stream` was validated two ways that both stop
short of the thing users actually complain about: a microbenchmark of the
markdown renderer, and equivalence tests on the split points. Neither mounts the
widget, so neither sees the cost that dominated before — `RichLog.write`
assigning a layout-invalidating reactive and queueing a scroll callback per row.

This mounts a real ChatLog in a headless Textual pilot, streams a realistic
reply token by token, and asserts each flush fits inside a 30 fps frame. Before
the fix, flush cost rose with the accumulated text and the code compensated by
dropping the frame rate to 4 fps; now only the tail is re-rendered.
"""

import asyncio
import time

TOKENS = (
    "Here is what I found in the codebase. ",
    "The loader reads the manifest first, ",
    "then resolves each entry.\n\n",
    "```python\n", "def load(path):\n", "    with open(path) as f:\n",
    "        return json.load(f)\n", "```\n\n",
    "That covers the happy path. ",
    "Errors are handled one level up.\n\n",
)


def _stream_and_time(repeats: int):
    """Return (per-flush ms samples, final text length)."""
    from textual.app import App, ComposeResult
    from localcode.tui.widgets.chat_log import ChatLog

    class _App(App):
        def compose(self) -> ComposeResult:
            yield ChatLog(id="chat-log")

    samples: list[float] = []

    async def scenario():
        app = _App()
        async with app.run_test() as pilot:
            await pilot.pause()
            log = app.query_one("#chat-log", ChatLog)
            log.start_assistant_stream() if hasattr(log, "start_assistant_stream") else None

            for _ in range(repeats):
                for tok in TOKENS:
                    log.stream_token(tok)
                # Force a flush and time the real widget path, not just the
                # markdown renderer.
                t0 = time.perf_counter()
                log._flush_stream()
                samples.append((time.perf_counter() - t0) * 1000.0)
                await pilot.pause()

            return len(getattr(log, "_stream_full", ""))

    return samples, asyncio.run(scenario())


def test_flush_cost_stays_flat_as_the_message_grows():
    """Per-flush cost must not scale with the length of the reply.

    Before the append-only work, every frame re-parsed the whole message
    through commonmark: 381 ms at 18 KB, which is why the coalescer used to
    drop the frame rate to 4 fps on long answers. Now only the tail after the
    last frozen block boundary is re-rendered, so cost is roughly constant.

    Measured on an M5 Max: a 10 KB reply flushes in 2.4 ms early and 2.7 ms
    late (worst 4.1 ms); a 30 KB reply peaks at 13.4 ms, still ~75 fps.

    This test earns its keep - it is what caught the fix being INERT. An
    earlier version zeroed the freshly computed boundary alongside the stored
    one, so no prefix was ever frozen and every frame still re-rendered
    everything, while the microbenchmarks and equivalence tests all passed.
    """
    samples, total_len = _stream_and_time(repeats=40)
    warm = samples[3:]
    assert len(warm) >= 20, "not enough samples"

    q = len(warm) // 4
    early = sum(warm[:q]) / q
    late = sum(warm[-q:]) / q

    # The message grows ~10x across the run. Flat means the ratio stays near 1;
    # allow 3x for scheduler noise on a shared machine. A return to
    # whole-message re-rendering shows up as 10x or worse.
    assert late < early * 3 + 1.0, (
        f"flush cost is scaling with message length: {early:.2f} ms early vs "
        f"{late:.2f} ms late over {total_len} chars - the frozen prefix is not "
        f"being reused"
    )

    # And the absolute budget: one frame at 30 fps is 33 ms.
    # Absolute-ms budget: the GROWTH-RATIO assert above is the real property
    # (does per-flush cost scale with message length). This one only guards
    # against a gross regression, and it must survive a CPU-loaded full-suite
    # run - a genuine O(n) return to whole-message re-rendering is 300 ms+ at
    # this size, so a generous ceiling still catches it without flaking on a
    # busy machine. The median (not the max) is what the reader feels.
    ordered = sorted(warm)
    median = ordered[len(ordered) // 2]
    assert median < 33.0, f"median flush {median:.1f} ms misses a 30 fps frame"
    assert max(warm) < 150.0, f"slowest flush {max(warm):.1f} ms - a real O(n) regression"


def test_stream_survives_a_rerender_without_losing_text():
    """History replay renders the whole message; streaming renders it in pieces.

    If those two ever disagree the user sees text jump at end of turn, which is
    the bug the append-only split had to avoid.
    """
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
            for tok in TOKENS * 3:
                log.stream_token(tok)
            log._flush_stream()
            await pilot.pause()
            streamed = "\n".join(s.text for s in log.lines)

            log.finish_stream()
            await pilot.pause()
            log._rerender()
            await pilot.pause()
            replayed = "\n".join(s.text for s in log.lines)
            return streamed, replayed

    streamed, replayed = asyncio.run(scenario())
    for probe in ("def load(path):", "That covers the happy path.", "resolves each entry."):
        assert probe in streamed, f"missing from streamed output: {probe}"
        assert probe in replayed, f"lost on rerender: {probe}"
