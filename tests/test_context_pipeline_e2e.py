"""End-to-end test for the context-reduction pipeline over a 25-turn session.

What this verifies
------------------
The three redaction layers (`_redact_old_write_args`,
`_redact_duplicate_reads`, `_compact_old_tool_results`) and the orchestrator
`_prepare_model_messages` are unit-tested individually. This test exercises
them together against a SYNTHETIC 25-turn coding-session message log that
mirrors the real failure pattern the user has been hitting:

  * many `write_file` calls with multi-kB content payloads
  * repeated `read_file` calls on the same paths across rounds
  * intermittent `bash` / `grep` calls with chunky output
  * occasional tool errors

Without redaction this kind of log grows unboundedly with every round —
each prior `write_file(content=<KB of code>)` re-ships in tool_calls every
subsequent round. With redaction the working set should stabilise: after
the first few writes redaction kicks in and the per-round byte size
should plateau, not climb linearly with turn number.

The test asserts:
  1. Pipeline never crashes / mutates input
  2. By turn 25 the redacted size is materially smaller than the unredacted
  3. Growth slows after redaction kicks in (sub-linear vs unredacted)
  4. Tool-call protocol invariants hold (every tool_role message has a
     matching tool_call_id from a prior assistant message)

Run: `python tests/test_context_pipeline_e2e.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode.agent import (
    _prepare_model_messages,
    _redact_old_write_args,
    _redact_duplicate_reads,
    _compact_old_tool_results,
    REDACT_KEEP_RECENT_WRITES,
    COMPACT_KEEP_RECENT_TOOL_RESULTS,
)


# ── Synthetic coding session ────────────────────────────────────────

_LARGE_PYTHON_FILE = """
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_config(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_output(data: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


class Pipeline:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.results: List[Any] = []

    def run(self) -> List[Any]:
        for item in self.config.get('items', []):
            self.results.append(self._process(item))
        return self.results

    def _process(self, item: Any) -> Any:
        return {'in': item, 'out': str(item).upper()}
""" * 5  # ~3 KB of "code"

_GREP_OUTPUT = "\n".join(
    f"src/file{i}.py:{j}: def function_{i}_{j}():" for i in range(20) for j in range(5)
)  # ~3 KB of grep output

_BASH_LS_OUTPUT = "\n".join(
    f"-rw-r--r--  1 u  s   {i*100} Apr 23 15:35 file{i}.py" for i in range(50)
)


def _tool_call(tc_id: str, name: str, args: dict) -> dict:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _assistant(tcs: list, content: str = "") -> dict:
    msg = {"role": "assistant", "content": content}
    if tcs:
        msg["tool_calls"] = tcs
    return msg


def _tool_result(tc_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def build_25_turn_session() -> list[dict]:
    """Replays a representative 25-turn coding session.

    Mix:
      * Turns 1-5: explore (read_file × 3 same files, grep, ls)
      * Turns 6-15: build (write_file × 2 per turn, occasional read)
      * Turns 16-20: edit (edit_file × 1 per turn, read of same file again)
      * Turns 21-25: test + fix (bash, write_file fixes, more reads)
    """
    messages: list[dict] = [
        {"role": "system", "content": "You are LocalCode. " * 200},  # ~3 KB system
        {"role": "user", "content": "build me a small Flask API for a TODO list"},
    ]
    counter = [0]

    def next_id() -> str:
        counter[0] += 1
        return f"tc-{counter[0]:03d}"

    files = ["app.py", "models.py", "routes.py", "config.py", "tests/test_app.py"]

    # Turns 1-5: explore (10 user msgs total since each turn = 1 user + 1+ assistant rounds)
    for t in range(1, 6):
        messages.append({"role": "user", "content": f"explore turn {t}"})
        # Read 1-2 files, run grep, run ls
        for f in files[:2]:
            tc_id = next_id()
            messages.append(_assistant([_tool_call(tc_id, "read_file", {"path": f})]))
            messages.append(_tool_result(tc_id, _LARGE_PYTHON_FILE))
        gid = next_id()
        messages.append(_assistant([_tool_call(gid, "grep", {"pattern": "def ", "path": "src/"})]))
        messages.append(_tool_result(gid, _GREP_OUTPUT))
        bid = next_id()
        messages.append(_assistant([_tool_call(bid, "bash", {"command": "ls -la"})]))
        messages.append(_tool_result(bid, _BASH_LS_OUTPUT))

    # Turns 6-15: write 2 files per turn, sometimes re-read what we wrote
    for t in range(6, 16):
        messages.append({"role": "user", "content": f"build turn {t}"})
        for f in files:
            wid = next_id()
            messages.append(_assistant([_tool_call(wid, "write_file",
                                                  {"path": f, "content": _LARGE_PYTHON_FILE})]))
            messages.append(_tool_result(wid, f"Written {f} ({_LARGE_PYTHON_FILE.count(chr(10))} lines)"))
        # Re-read app.py to verify
        rid = next_id()
        messages.append(_assistant([_tool_call(rid, "read_file", {"path": "app.py"})]))
        messages.append(_tool_result(rid, _LARGE_PYTHON_FILE))

    # Turns 16-20: edits + re-reads
    for t in range(16, 21):
        messages.append({"role": "user", "content": f"edit turn {t}"})
        eid = next_id()
        messages.append(_assistant([_tool_call(eid, "edit_file", {
            "path": "app.py",
            "old_string": "def parse_config",
            "new_string": _LARGE_PYTHON_FILE[:500],
        })]))
        messages.append(_tool_result(eid, "edited app.py"))
        rid = next_id()
        messages.append(_assistant([_tool_call(rid, "read_file", {"path": "app.py"})]))
        messages.append(_tool_result(rid, _LARGE_PYTHON_FILE))

    # Turns 21-25: test + fix
    for t in range(21, 26):
        messages.append({"role": "user", "content": f"test turn {t}"})
        bid = next_id()
        messages.append(_assistant([_tool_call(bid, "bash", {"command": "pytest -q"})]))
        messages.append(_tool_result(bid, _BASH_LS_OUTPUT))
        wid = next_id()
        messages.append(_assistant([_tool_call(wid, "write_file",
                                               {"path": "tests/test_app.py",
                                                "content": _LARGE_PYTHON_FILE})]))
        messages.append(_tool_result(wid, "Written tests/test_app.py"))

    return messages


# ── Assertions ──────────────────────────────────────────────────────


def _bytes(messages: list[dict]) -> int:
    return len(json.dumps(messages))


def _validate_protocol(messages: list[dict]) -> None:
    """Every tool-role message must have a tool_call_id that appears in a
    prior assistant message's tool_calls. Without this, the OpenAI API
    rejects the request with "tool message without preceding tool_call."
    """
    valid_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if tc.get("id"):
                    valid_ids.add(tc["id"])
        elif m.get("role") == "tool":
            tc_id = m.get("tool_call_id")
            assert tc_id in valid_ids, (
                f"orphan tool result with id={tc_id!r} — no matching tool_call"
            )


def main() -> int:
    print("Building 25-turn synthetic session...")
    messages = build_25_turn_session()
    print(f"  messages: {len(messages)}")
    print(f"  raw size: {_bytes(messages):>10} bytes")
    print()

    # Sanity 1: pipeline doesn't blow up + doesn't mutate input
    snapshot_before = json.dumps(messages)
    reduced = _prepare_model_messages(messages)
    snapshot_after = json.dumps(messages)
    assert snapshot_before == snapshot_after, "_prepare_model_messages mutated input"
    print("✓ pipeline runs without mutating input")

    # Sanity 2: protocol invariants
    _validate_protocol(messages)
    _validate_protocol(reduced)
    print("✓ tool_call_id chain intact in both raw + reduced")

    # Per-pass deltas
    after_writes = _redact_old_write_args(messages)
    saved_writes = _bytes(messages) - _bytes(after_writes)
    after_reads = _redact_duplicate_reads(after_writes)
    saved_reads = _bytes(after_writes) - _bytes(after_reads)
    after_tools = _compact_old_tool_results(after_reads)
    saved_tools = _bytes(after_reads) - _bytes(after_tools)
    total_saved = saved_writes + saved_reads + saved_tools
    pct = total_saved * 100 / _bytes(messages)
    print()
    print(f"  bytes saved by write-arg redaction:   {saved_writes:>10}")
    print(f"  bytes saved by dup-read stubbing:     {saved_reads:>10}")
    print(f"  bytes saved by tool-result aging:     {saved_tools:>10}")
    print(f"  total saved:                          {total_saved:>10}  ({pct:.1f}%)")
    print(f"  reduced size:                         {_bytes(reduced):>10} bytes")
    print()

    # Hard assertion: by turn 25 the reduction must be material.
    # Threshold: ≥40% saved is the bar — anything less means the layered
    # redaction isn't pulling its weight on a realistic coding session.
    assert pct >= 40.0, (
        f"redaction pipeline too weak: only {pct:.1f}% saved on a 25-turn coding session "
        f"(expected ≥40%). Pipeline isn't pulling its weight."
    )
    print(f"✓ pipeline saves ≥40% of context bytes on a realistic 25-turn session")

    # Bounded growth: simulate progressive turns and check that the
    # reduced size stops growing linearly once redaction kicks in.
    print()
    print("Bounded-growth check (per-round size after redaction):")
    print(f"  {'turn':>5}  {'raw':>10}  {'reduced':>10}  {'ratio':>6}")
    bench_messages: list[dict] = [
        {"role": "system", "content": "You are LocalCode. " * 200},
        {"role": "user", "content": "build"},
    ]
    cnt = [0]
    def _id(): cnt[0] += 1; return f"b-{cnt[0]:03d}"
    sizes = []
    for t in range(1, 26):
        # Add one user msg + one write_file + one tool_result per turn
        bench_messages.append({"role": "user", "content": f"turn {t}"})
        wid = _id()
        bench_messages.append(_assistant([_tool_call(wid, "write_file",
            {"path": f"f{t}.py", "content": _LARGE_PYTHON_FILE})]))
        bench_messages.append(_tool_result(wid, f"Written f{t}.py"))
        raw = _bytes(bench_messages)
        red = _bytes(_prepare_model_messages(bench_messages))
        sizes.append((t, raw, red))
        if t in (1, 5, 10, 15, 20, 25):
            print(f"  {t:>5}  {raw:>10}  {red:>10}  {red/raw:>5.2f}")

    # After redaction kicks in (turn > REDACT_KEEP_RECENT_WRITES), per-turn
    # growth of the REDUCED size should be sub-linear vs growth of RAW size.
    raw_growth = sizes[-1][1] - sizes[REDACT_KEEP_RECENT_WRITES][1]
    red_growth = sizes[-1][2] - sizes[REDACT_KEEP_RECENT_WRITES][2]
    growth_ratio = red_growth / max(1, raw_growth)
    print()
    print(f"  raw growth (turn {REDACT_KEEP_RECENT_WRITES+1}..25):     {raw_growth:>10} bytes")
    print(f"  reduced growth (same window):  {red_growth:>10} bytes")
    print(f"  growth ratio:                  {growth_ratio:.3f}")
    assert growth_ratio < 0.4, (
        f"reduced size still growing too fast: ratio {growth_ratio:.3f} (expected < 0.4). "
        f"Redaction isn't bounding per-turn growth."
    )
    print(f"✓ reduced size grows < 40% as fast as raw — redaction is bounding growth")

    print()
    print("=" * 60)
    print(f"PASS — 25-turn pipeline test")
    print(f"  saved {total_saved} bytes ({pct:.1f}%) over a 25-turn session")
    print(f"  per-turn growth bounded to {growth_ratio*100:.1f}% of unredacted")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
