"""Tests for the read-before-edit guard, typography-tolerant fuzzy matching,
multi_edit clobber guard, and the context aging whitelist.

Self-contained: imports the tool modules directly (not through
`localcode.app`, which pulls optional network deps) so it runs in the minimal
CI environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode.tools import read_state
from localcode.tools.base import ToolContext
from localcode.tools.edit_file import execute as edit_file
from localcode.tools.write_file import execute as write_file
from localcode.tools.multi_edit import execute as multi_edit
from localcode.tools.read_file import execute as read_file


class _Out:
    def print_info(self, *a, **k):
        pass


class _App:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(app=_App(tmp_path), out=_Out())


def _read(ctx: ToolContext, rel: str, **kw) -> str:
    return read_file(ctx, {"path": rel, **kw})


# ── 1. Read-before-edit staleness guard ─────────────────────────────


def test_edit_without_any_read_is_allowed_when_session_unarmed(tmp_path: Path) -> None:
    # A session that has never read anything has no freshness baseline, so the
    # guard stays out of the way (keeps direct/unit callers working).
    (tmp_path / "a.py").write_text("alpha\n")
    ctx = _ctx(tmp_path)
    assert not read_state.is_armed(ctx.app)
    out = edit_file(ctx, {"path": "a.py", "old_string": "alpha", "new_string": "beta"})
    assert "beta" in (tmp_path / "a.py").read_text()
    assert "Error" not in out.splitlines()[0]


def test_edit_unread_file_refused_once_armed(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\n")
    (tmp_path / "b.py").write_text("bravo\n")
    ctx = _ctx(tmp_path)
    _read(ctx, "a.py")  # arms the session
    out = edit_file(ctx, {"path": "b.py", "old_string": "bravo", "new_string": "x"})
    assert out.startswith("REJECTED:")
    assert "read" in out.lower() and "b.py" in out
    assert (tmp_path / "b.py").read_text() == "bravo\n"  # unchanged


def test_partial_read_does_not_satisfy_guard(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("l1\nl2\nl3\nl4\n")
    ctx = _ctx(tmp_path)
    _read(ctx, "a.py", offset=1)  # partial (offset > 0)
    out = edit_file(ctx, {"path": "a.py", "old_string": "l2", "new_string": "X"})
    assert out.startswith("REJECTED:")
    assert "part" in out.lower()


def test_full_read_then_edit_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\n")
    ctx = _ctx(tmp_path)
    _read(ctx, "a.py")
    out = edit_file(ctx, {"path": "a.py", "old_string": "alpha", "new_string": "beta"})
    assert "Error" not in out.splitlines()[0]
    assert (tmp_path / "a.py").read_text() == "beta\n"


def test_stale_file_refused(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("alpha\n")
    ctx = _ctx(tmp_path)
    _read(ctx, "a.py")
    # External change: different content + advanced mtime.
    p.write_text("ALPHA CHANGED\n")
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 10))
    out = edit_file(ctx, {"path": "a.py", "old_string": "ALPHA", "new_string": "x"})
    assert out.startswith("REJECTED:")
    assert "changed on disk" in out


def test_content_equality_fallback_allows_touch(tmp_path: Path) -> None:
    # mtime bumped but bytes identical (formatter/AV touch) → not a false reject.
    p = tmp_path / "a.py"
    p.write_text("alpha\n")
    ctx = _ctx(tmp_path)
    _read(ctx, "a.py")
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 10))  # mtime only, same bytes
    out = edit_file(ctx, {"path": "a.py", "old_string": "alpha", "new_string": "beta"})
    assert "Error" not in out.splitlines()[0]
    assert (tmp_path / "a.py").read_text() == "beta\n"


def test_model_own_write_then_edit_not_blocked(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _read(ctx, "seed.py") if (tmp_path / "seed.py").write_text("x\n") else None  # arm
    # write_file creates a new file; the model must be able to edit it after.
    write_file(ctx, {"path": "new.py", "content": "one = 1\ntwo = 2\n"})
    out = edit_file(ctx, {"path": "new.py", "old_string": "one = 1", "new_string": "one = 111"})
    assert "Error" not in out.splitlines()[0]
    assert "one = 111" in (tmp_path / "new.py").read_text()


def test_edit_then_reedit_same_file_not_blocked(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("a = 1\nb = 2\n")
    ctx = _ctx(tmp_path)
    _read(ctx, "a.py")
    edit_file(ctx, {"path": "a.py", "old_string": "a = 1", "new_string": "a = 11"})
    out = edit_file(ctx, {"path": "a.py", "old_string": "b = 2", "new_string": "b = 22"})
    assert "Error" not in out.splitlines()[0]
    assert (tmp_path / "a.py").read_text() == "a = 11\nb = 22\n"


def test_write_overwrite_of_stale_file_refused(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("original\n")
    ctx = _ctx(tmp_path)
    _read(ctx, "a.py")
    p.write_text("changed by someone else\n")
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 10))
    out = write_file(ctx, {"path": "a.py", "content": "my full rewrite\n"})
    assert out.startswith("REJECTED:")
    assert "clobber" in out
    assert (tmp_path / "a.py").read_text() == "changed by someone else\n"


def test_write_overwrite_of_unread_file_allowed(tmp_path: Path) -> None:
    # write_file does NOT require a prior read (it is the full-rewrite tool).
    (tmp_path / "seed.py").write_text("seed\n")
    (tmp_path / "a.py").write_text("original\n")
    ctx = _ctx(tmp_path)
    _read(ctx, "seed.py")  # arm, but do NOT read a.py
    out = write_file(ctx, {"path": "a.py", "content": "rewritten\n"})
    assert "Error" not in out.splitlines()[0]
    assert (tmp_path / "a.py").read_text() == "rewritten\n"


# ── 2. Typography-tolerant fuzzy matching ───────────────────────────


def test_straight_quotes_match_curly_and_preserve_typography(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text('msg = “hello”\n')  # curly double quotes in file
    ctx = _ctx(tmp_path)
    out = edit_file(ctx, {
        "path": "a.py",
        "old_string": 'msg = "hello"',   # model typed STRAIGHT quotes
        "new_string": 'msg = "world"',
    })
    assert "Error" not in out.splitlines()[0]
    # Replacement preserves the file's curly typography.
    assert (tmp_path / "a.py").read_text() == 'msg = “world”\n'


def test_curly_single_quote_contraction_preserved(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("s = ‘hi’\n")  # curly single quotes
    ctx = _ctx(tmp_path)
    out = edit_file(ctx, {
        "path": "a.py",
        "old_string": "s = 'hi'",
        "new_string": "s = 'yo'",
    })
    assert "Error" not in out.splitlines()[0]
    assert (tmp_path / "a.py").read_text() == "s = ‘yo’\n"


def test_unicode_dash_tolerated(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("r = a — b\n")  # em dash in file
    ctx = _ctx(tmp_path)
    out = edit_file(ctx, {
        "path": "a.py",
        "old_string": "r = a - b",  # ascii hyphen
        "new_string": "r = a + b",
    })
    assert "Error" not in out.splitlines()[0]
    assert (tmp_path / "a.py").read_text() == "r = a + b\n"


# ── 3. multi_edit clobber guard + no-op detection ───────────────────


def test_multi_edit_clobber_guard(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("AAA\nBBB\n")
    ctx = _ctx(tmp_path)
    out = multi_edit(ctx, {
        "path": "a.py",
        "edits": [
            {"old_string": "AAA", "new_string": "XXX BBB YYY"},  # inserts BBB
            {"old_string": "BBB", "new_string": "ZZZ"},          # anchors on it
        ],
    })
    assert "substring of edit 1's new_string" in out
    assert "applied 0/2" in out
    assert (tmp_path / "a.py").read_text() == "AAA\nBBB\n"  # atomic: unchanged


def test_multi_edit_noop_single_edit_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("AAA\nBBB\n")
    ctx = _ctx(tmp_path)
    out = multi_edit(ctx, {
        "path": "a.py",
        "edits": [{"old_string": "AAA", "new_string": "AAA"}],
    })
    assert "no-op" in out.lower()
    assert "applied 0/1" in out


def test_multi_edit_independent_edits_still_apply(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("AAA\nBBB\n")
    ctx = _ctx(tmp_path)
    out = multi_edit(ctx, {
        "path": "a.py",
        "edits": [
            {"old_string": "AAA", "new_string": "111"},
            {"old_string": "BBB", "new_string": "222"},
        ],
    })
    assert "Applied 2/2" in out
    assert (tmp_path / "a.py").read_text() == "111\n222\n"


# ── 4. Context aging whitelist (write bodies preserved) ─────────────


def _tc(tc_id, name, args):
    import json
    return {"id": tc_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_hot_reads_protected_writes_preserved_old_bash_aged() -> None:
    """Aging preserves write diffs AND the latest read of each hot file (so the
    model doesn't re-read files it already has), while still aging older
    non-read replayable output (e.g. bash) to bound context."""
    from localcode.agent.context import _compact_old_tool_results
    big = "x" * 600  # > COMPACT_MIN_CONTENT_CHARS
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "", "tool_calls": [_tc("b1", "bash", {})]},
        {"role": "tool", "tool_call_id": "b1", "content": "BASH-OLD " + big},
        {"role": "assistant", "content": "", "tool_calls": [_tc("r1", "read_file", {"path": "a.py"})]},
        {"role": "tool", "tool_call_id": "r1", "content": "READFILE-A " + big},
        {"role": "assistant", "content": "", "tool_calls": [_tc("w1", "write_file", {"path": "b.py"})]},
        {"role": "tool", "tool_call_id": "w1", "content": "WRITE-DIFF " + big},
        {"role": "assistant", "content": "", "tool_calls": [_tc("b2", "bash", {})]},
        {"role": "tool", "tool_call_id": "b2", "content": "BASH-RECENT " + big},
        {"role": "assistant", "content": "", "tool_calls": [_tc("r2", "read_file", {"path": "c.py"})]},
        {"role": "tool", "tool_call_id": "r2", "content": "READFILE-C " + big},
    ]
    out = _compact_old_tool_results(msgs, keep_recent=1)
    by_id = {m.get("tool_call_id"): m for m in out if m.get("role") == "tool"}
    # Old bash output (replayable, NOT a hot read) is aged/shrunk.
    assert ("x" * 600) not in by_id["b1"]["content"]
    assert len(by_id["b1"]["content"]) < len("BASH-OLD " + big)
    # The latest read of each hot file is PROTECTED from aging (the re-read fix).
    assert by_id["r1"]["content"] == "READFILE-A " + big
    assert by_id["r2"]["content"] == "READFILE-C " + big
    # Write result (the diff of what the model wrote) is NEVER aged.
    assert by_id["w1"]["content"] == "WRITE-DIFF " + big
    # Most-recent replayable result kept verbatim (keep_recent=1).
    assert by_id["b2"]["content"] == "BASH-RECENT " + big


def test_redact_old_write_args_never_strips_bodies() -> None:
    from localcode.agent.context import _redact_old_write_args
    body = "print('hello')\n" * 200
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(5):
        msgs.append({"role": "user", "content": f"req {i}"})
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [_tc(f"w{i}", "write_file", {"path": f"f{i}.py", "content": body})]})
        msgs.append({"role": "tool", "tool_call_id": f"w{i}", "content": f"Created f{i}.py"})
    out = _redact_old_write_args(msgs)
    import json
    dumped = json.dumps(out)
    # Every write body must survive — none redacted.
    assert dumped.count("print('hello')") == 5 * 200
    assert "REDACTED" not in dumped


def test_guard_steers_render_calm_not_red():
    """Read-before-edit / staleness guards are deliberate STEERS (the model
    re-reads and recovers), not failures. They must use the REJECTED prefix so
    the loop renders them as a calm neutral note, not an alarming red error
    block (the pre-marketing UX wart where a blocked write flashed red)."""
    from localcode.tools import read_state as rs
    import inspect
    src = inspect.getsource(rs.guard_edit) + inspect.getsource(rs.guard_overwrite)
    # No guard steer may start with a bare "Error:" (that path renders red).
    assert 'f"Error:' not in src, "guard steers must not use the red Error: prefix"
    # They must use REJECTED so loop.py's is_rejected calm-render path catches them.
    assert 'REJECTED:' in src
