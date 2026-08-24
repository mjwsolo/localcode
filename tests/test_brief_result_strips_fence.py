"""The tool-result summary must never show the injection-defense fence.

Live rendering strips the <UNTRUSTED_DATA source="..."> wrapper before
summarizing, but a history re-render recomputes the summary from the raw
stored result - which leaked `<UNTRUSTED_DATA source="$ node -v…` onto the
tool-done row (caught in a real qwen38 build). _brief_result now unwraps first.
"""

from __future__ import annotations

from localcode.agent.helpers import _brief_result
from localcode.injection_defense import wrap_untrusted


def test_brief_result_strips_untrusted_fence():
    wrapped = wrap_untrusted(
        "v20.11.0\n10.9.0\ntotal 0\na\nb\nc\nd\ne\n",
        source="$ node -v && npm -v && ls -la /Users/x/proj",
    )
    brief = _brief_result("bash", wrapped)
    assert "UNTRUSTED_DATA" not in brief
    assert "v20.11.0" in brief


def test_brief_result_clean_input_unchanged():
    brief = _brief_result("bash", "hello\nworld\n")
    assert "UNTRUSTED_DATA" not in brief
    assert "hello" in brief
