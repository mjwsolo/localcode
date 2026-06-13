"""Tests for conversation compaction — RAM-tiered summary + window scaling.

The behaviour under test (added when compaction was made per-machine):
  * Big-RAM machines spend a model generation for a rich summary.
  * Small-RAM machines (< LLM_SUMMARY_MIN_RAM_GB) use an instant
    deterministic summary — NO model call (no mid-task stall, no weak
    small-model summary).
  * If the LLM summary call fails or returns nothing, ANY machine falls
    back to the deterministic summary rather than corrupting history.
  * keep_recent_tokens scales with the context window so a 256K window
    preserves far more recent history than a 64K one.
"""
from __future__ import annotations

from localcode import compaction
from localcode.compaction import (
    LLM_SUMMARY_MIN_RAM_GB,
    KEEP_RECENT_TOKENS_DEFAULT,
    KEEP_RECENT_TOKENS_MAX,
    _keep_recent_for_window,
    compact,
)


class _RecordingRuntime:
    """Fake runtime that records whether chat_once was called."""

    def __init__(self, reply: str | None = "## Goal\nrich llm summary"):
        self.reply = reply
        self.called = 0

    def chat_once(self, messages, tools=None, think=False, num_predict=None):
        self.called += 1
        return {"message": {"content": self.reply or ""}}


def _long_history(n: int = 60) -> list[dict]:
    """A conversation long enough that _split_at_keep_recent leaves an
    old slice to summarize."""
    msgs: list[dict] = [{"role": "system", "content": "system prompt"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"please edit file_{i}.py " + "x" * 400})
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "file_%d.py", "content": "%s"}' % (i, "y" * 400),
                        },
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok " + "z" * 400})
    return msgs


# Force a small keep-recent so _long_history() always leaves an old slice
# to summarize, independent of window scaling (that's tested separately).
_KEEP = 2000


def test_small_ram_uses_deterministic_no_model_call():
    rt = _RecordingRuntime()
    msgs = _long_history()
    out = compact(msgs, rt, context_window=65536, ram_gb=16, keep_recent_tokens=_KEEP)
    assert rt.called == 0, "small RAM must NOT spend a model generation on the summary"
    # A summary memo was still inserted.
    memos = [m for m in out if m.get("role") == "system" and "Prior conversation summary" in (m.get("content") or "")]
    assert memos, "deterministic summary memo should be present"
    # The deterministic summary preserves files touched.
    assert "file_" in memos[0]["content"]


def test_big_ram_spends_llm_generation():
    rt = _RecordingRuntime()
    msgs = _long_history()
    out = compact(msgs, rt, context_window=262144, ram_gb=128, keep_recent_tokens=_KEEP)
    assert rt.called == 1, "capable RAM should call the model once for the summary"
    memo = next(m for m in out if "Prior conversation summary" in (m.get("content") or ""))
    assert "rich llm summary" in memo["content"]


def test_llm_failure_falls_back_to_deterministic():
    # Big RAM, but the model returns empty — must not corrupt history.
    rt = _RecordingRuntime(reply="")
    msgs = _long_history()
    out = compact(msgs, rt, context_window=262144, ram_gb=128, keep_recent_tokens=_KEEP)
    assert rt.called == 1
    memo = next((m for m in out if "Prior conversation summary" in (m.get("content") or "")), None)
    assert memo is not None, "must fall back to deterministic summary, not bail"
    assert "file_" in memo["content"]


def test_threshold_boundary_is_inclusive():
    rt = _RecordingRuntime()
    compact(_long_history(), rt, context_window=131072, ram_gb=LLM_SUMMARY_MIN_RAM_GB, keep_recent_tokens=_KEEP)
    assert rt.called == 1, "ram == LLM_SUMMARY_MIN_RAM_GB should use the LLM path"

    rt2 = _RecordingRuntime()
    compact(_long_history(), rt2, context_window=131072, ram_gb=LLM_SUMMARY_MIN_RAM_GB - 1, keep_recent_tokens=_KEEP)
    assert rt2.called == 0, "just below threshold should be deterministic"


def test_ram_none_defaults_to_llm_for_backcompat():
    rt = _RecordingRuntime()
    compact(_long_history(), rt, context_window=131072, keep_recent_tokens=_KEEP)  # no ram_gb
    assert rt.called == 1, "ram_gb=None preserves the original LLM behaviour"


def test_keep_recent_scales_with_window():
    tiny = _keep_recent_for_window(16384)
    big = _keep_recent_for_window(262144)
    assert tiny == KEEP_RECENT_TOKENS_DEFAULT, "tiny window keeps only the floor"
    assert big > tiny, "big window keeps more recent history verbatim"
    assert big <= KEEP_RECENT_TOKENS_MAX, "but is capped"
    # Unknown/zero window → floor.
    assert _keep_recent_for_window(0) == KEEP_RECENT_TOKENS_DEFAULT


def test_nothing_to_summarize_returns_unchanged():
    rt = _RecordingRuntime()
    short = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = compact(short, rt, context_window=262144, ram_gb=128)
    assert out is short, "no old slice → return the input unchanged"
    assert rt.called == 0
