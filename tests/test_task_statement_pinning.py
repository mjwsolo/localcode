"""Regression: compaction must never lose the user's task statement.

Observed live on a long agentic build: after compaction the model said
"I need to understand what the actual task is — it seems like building an
Anki clone based on the directory name". The original request had been
summarized into a one-line ledger. Both compactors must carry the task
statement forward VERBATIM.
"""
from localcode.agent.context import _microcompact_for_prompt_budget
from localcode.auto_compact import auto_compact
from localcode.app import _prompt_names_explicit_path

TASK = "Build me a local-first Anki clone as a single-page web app with FSRS scheduling"


def _long_history(task_first: bool = True) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": "You are LocalCode."}]
    if task_first:
        msgs.append({"role": "user", "content": TASK})
    for i in range(30):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "write_file", "arguments": f'{{"path": "src/f{i}.ts", "content": "{"x" * 3000}"}}'}}
        ]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"wrote src/f{i}.ts"})
    return msgs


def test_microcompact_keeps_task_statement_verbatim() -> None:
    msgs = _long_history()
    out = _microcompact_for_prompt_budget(msgs, target_bytes=8_000)
    assert len(out) < len(msgs)  # it actually compacted
    contents = [str(m.get("content", "")) for m in out]
    assert any(TASK in c for c in contents), "task statement was compacted away"
    # And it must be the verbatim user message, not a summary mention.
    assert any(m.get("role") == "user" and m.get("content") == TASK for m in out)


def test_auto_compact_keeps_task_statement_verbatim() -> None:
    msgs = _long_history()
    out, summary = auto_compact(msgs, max_chars=10_000, keep_recent=6)
    assert len(out) < len(msgs)
    assert any(m.get("role") == "user" and m.get("content") == TASK for m in out), \
        "auto_compact summarized away the active task statement"


def test_auto_compact_task_in_recent_tail_not_duplicated() -> None:
    # When the task statement is already inside the kept-recent tail, it must
    # not be duplicated at the head.
    msgs = _long_history(task_first=False)
    msgs.append({"role": "user", "content": TASK})
    msgs.append({"role": "assistant", "content": "on it"})
    out, _ = auto_compact(msgs, max_chars=10_000, keep_recent=6)
    assert sum(1 for m in out if m.get("content") == TASK) == 1


def test_prompt_path_detection_guards_home_anchor() -> None:
    # User names a location → anchor must not fire.
    assert _prompt_names_explicit_path("build it in ~/Desktop/Github/localcode_test/Anki")
    assert _prompt_names_explicit_path("put the app in ./projects/anki")
    assert _prompt_names_explicit_path("use /Users/me/dev as the target")
    # No location named → anchor may fire. URLs are not paths.
    assert not _prompt_names_explicit_path("build me an anki clone as a web app")
    assert not _prompt_names_explicit_path("fetch https://example.com/docs and build from spec")
