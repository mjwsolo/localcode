"""localcode Checks: deterministic assertions over the real UI runtime driven
against canned servers (no model). Each test names the quality it protects.

Run:  python -m pytest evals/checks -q   (localcode importable; see run_checks.py)
Env:  LOCALCODE_CHECK_REPO (default: this repo), a checkout with src/localcode/bin/localcode-ui.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("LOCALCODE_CHECK_REPO") or HERE.parents[1]).resolve()
DRIVER = HERE / "run_checks.py"

# Budgets. Tight enough to catch bloat, loose enough for normal drift. The
# system prompt is constant for a whole session by design (0.5.6), so one
# budget covers every turn.
SYSTEM_CHARS = 12_000                     # today: 8,882 (planning rules included from turn 1)
TOOL_SCHEMA_BYTES = 9_000                 # today: 7,142 for 10 tools (lean definitions, 0.5.10)


def _run(slots: int, plugin: str | None = None) -> dict:
    out = Path(tempfile.mkdtemp(prefix=f"lc-checks-{slots}-"))
    cmd = [sys.executable, str(DRIVER), str(REPO), str(out), "--slots", str(slots)] + (["--plugin", plugin] if plugin else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    reqs = []
    for f in sorted(out.glob("req-*.json")):
        b = json.loads(f.read_text())
        msgs = b["messages"]
        system = "\n".join(m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]) for m in msgs if m["role"] == "system")
        tools = b.get("tools") or []
        reqs.append({
            "name": f.name, "body": b, "system": system, "tools": tools,
            "sys_sha": hashlib.sha1(system.encode()).hexdigest(), "tools_sha": hashlib.sha1(json.dumps(tools, sort_keys=True).encode()).hexdigest(),
            "users": sum(1 for m in msgs if m["role"] == "user"), "last": msgs[-1]["role"], "id_slot": b.get("id_slot"),
        })
    return {"out": out, "summary": json.loads((out / "summary.json").read_text()), "reqs": reqs,
            "ui_log": (out / "ui.log").read_text(errors="replace"), "stderr": "".join(p.read_text(errors="replace") for p in out.glob("stderr-*.txt"))}


@pytest.fixture(scope="module")
def four():
    return _run(4)


@pytest.fixture(scope="module")
def one():
    return _run(1)


def _agent_requests(run):
    return [r for r in run["reqs"] if r["tools"]]


# ── Reliability: the conversation ran ────────────────────────────────

def test_both_turns_complete_and_tools_execute(four):
    assert [t["rc"] for t in four["summary"]["turns"]] == [0, 0]
    agent = _agent_requests(four)
    assert len(agent) == 4, [r["name"] for r in four["reqs"]]
    # the read tool ran in turn 1 and todowrite ran in turn 2: each turn has a request whose last message is a tool result
    assert [r["last"] for r in agent] == ["user", "tool", "user", "tool"]
    assert "auto-rejecting" not in four["stderr"], four["stderr"][-500:]
    assert four["summary"]["workspace_files_changed"] == []


def test_no_runtime_errors(four):
    errors = [l for l in four["ui_log"].splitlines() if "level=ERROR" in l]
    assert errors == [], errors[:3]
    assert "invalid user part" not in four["ui_log"]
    assert "InvalidDurableEvent" not in four["ui_log"]


# ── Efficiency: the cached prefix stays put ──────────────────────────

def test_system_prompt_changes_at_most_once_per_session(four):
    agent = _agent_requests(four)
    shas = [r["sys_sha"] for r in agent]
    changes = sum(1 for a, b in zip(shas, shas[1:]) if a != b)
    assert changes <= 1, f"system prompt changed {changes} times across the session: {shas}"
    if changes == 1:
        # the one allowed change is the workspace switch, right after the first tool call
        assert shas[0] != shas[1] and shas[1] == shas[2] == shas[3], shas


def test_tool_schemas_identical_across_session(four):
    agent = _agent_requests(four)
    assert len({r["tools_sha"] for r in agent}) == 1


def test_second_turn_starts_with_the_first_turns_prefix(four):
    agent = _agent_requests(four)
    assert agent[2]["sys_sha"] == agent[1]["sys_sha"], "turn 2 did not reuse turn 1's system prompt"


def test_prompt_budgets(four):
    agent = _agent_requests(four)
    assert len(agent[0]["system"]) <= SYSTEM_CHARS, len(agent[0]["system"])
    assert len(agent[-1]["system"]) <= SYSTEM_CHARS, len(agent[-1]["system"])
    assert len(json.dumps(agent[0]["tools"])) <= TOOL_SCHEMA_BYTES, len(json.dumps(agent[0]["tools"]))


def test_side_requests_pinned_to_last_slot(four, one):
    side4 = [r for r in four["reqs"] if not r["tools"]]
    assert side4 and all(r["id_slot"] == 3 for r in side4), [(r["name"], r["id_slot"]) for r in side4]
    assert all(r["id_slot"] is None for r in _agent_requests(four))
    assert all(r["id_slot"] is None for r in one["reqs"]), [(r["name"], r["id_slot"]) for r in one["reqs"]]


# ── Fidelity: the model is told the truth about itself ───────────────

def test_model_name_in_system_prompt(four):
    agent = _agent_requests(four)
    assert "powered by the model named canned-model" in agent[0]["system"]
    assert "__pending__" not in agent[0]["system"]


def test_no_todo_list_in_system_prompt(four):
    assert all("YOUR OPEN TODOS" not in r["system"] for r in four["reqs"])


# ── The check can fail: a plugin that stamps the time into the system prompt ──

def test_prefix_check_detects_a_changing_prompt():
    src = (REPO / "src/localcode/ui/plugin/localcode.ts").read_text()
    anchor = '"experimental.chat.system.transform": async (input, output) => {'
    assert anchor in src
    mutated = src.replace(anchor, anchor + '\n      output.system.push("stamp " + String(Date.now()));', 1)
    tmp = Path(tempfile.mkdtemp(prefix="lc-mutant-")) / "localcode.ts"
    tmp.write_text(mutated)
    run = _run(4, plugin=str(tmp))
    shas = [r["sys_sha"] for r in _agent_requests(run)]
    assert len(set(shas)) == len(shas), "mutant went undetected: prefixes were identical"
