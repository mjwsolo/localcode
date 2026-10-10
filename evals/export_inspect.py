#!/usr/bin/env python3
"""Export eval runs to Inspect AI logs so `inspect view` can browse them.

    python3 evals/export_inspect.py [runs_dir ...] [--out evals/runs/inspect] [--view]

Each run directory (results.jsonl + trials/<task>-<n>/events-turn*.jsonl, as
written by run_tasks.py) becomes one Inspect eval log: one sample per trial,
the transcript rebuilt from the saved UI events (assistant text, tool calls,
tool results), the grader's verdict as the score, and the runner's metrics
(steps, wall time, tool-call validity, re-read tokens) as sample metadata.
Nothing is sent anywhere; `inspect view` serves the logs from localhost.

Requires `pip install inspect-ai` (only here, not in LocalCode itself).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from inspect_ai.log import (EvalConfig, EvalDataset, EvalLog, EvalMetric, EvalResults, EvalSample,
                                EvalSampleScore, EvalScore, EvalSpec, EvalStats, write_eval_log)
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
    from inspect_ai.tool import ToolCall
except ImportError:
    sys.exit("inspect-ai is not installed: pip install inspect-ai")

HERE = Path(__file__).resolve().parent


def turns_of(task_dir: Path) -> list[str]:
    text = (task_dir / "instruction.md").read_text() if (task_dir / "instruction.md").is_file() else ""
    if "## Turn " in text:
        parts = [p.split("\n", 1)[1].strip() for p in text.split("## Turn ")[1:]]
        return [p for p in parts if p]
    return [text.strip()] if text.strip() else ["(instruction missing)"]


def messages_from_events(turns: list[str], trial_dir: Path) -> list:
    """Rebuild a chat transcript from the per-turn UI event streams."""
    msgs: list = []
    i = 0
    while True:
        f = trial_dir / f"events-turn{i + 1}.jsonl"
        if not f.is_file():
            break
        msgs.append(ChatMessageUser(content=turns[i] if i < len(turns) else f"(turn {i + 1})"))
        text_parts: list[str] = []
        pending_calls: list[ToolCall] = []
        results: list[tuple[str, str, str, bool]] = []   # (id, tool, output, error)
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = e.get("part") or {}
            t = e.get("type")
            if t == "text":
                text_parts.append(str(p.get("text") or e.get("text") or ""))
            elif t == "tool_use":
                st = p.get("state") or {}
                cid = str(p.get("callID") or p.get("id") or f"call{len(pending_calls)}")
                tool = str(p.get("tool") or "tool")
                pending_calls.append(ToolCall(id=cid, function=tool, arguments=st.get("input") or {}))
                err = st.get("status") == "error"
                out = str(st.get("error") if err else (st.get("output") or ""))
                results.append((cid, tool, out[:4000], err))
        # One assistant turn holding every call, then the tool results, then any text.
        if pending_calls:
            msgs.append(ChatMessageAssistant(content="", tool_calls=pending_calls))
            for cid, tool, out, err in results:
                msgs.append(ChatMessageTool(content=out, tool_call_id=cid, function=tool))
        msgs.append(ChatMessageAssistant(content="\n".join(text_parts).strip() or "(no text)"))
        i += 1
    return msgs


def export_run(run: Path, out_dir: Path, tasks_dir: Path) -> Path | None:
    rows = []
    for line in (run / "results.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        return None
    model = rows[0].get("model", "unknown")
    label = rows[0].get("label", run.name)
    samples = []
    for r in rows:
        task_dir = tasks_dir / r["task"]
        turns = turns_of(task_dir)
        trial_dir = run / "trials" / f"{r['task']}-{r['trial']}"
        msgs = messages_from_events(turns, trial_dir) if trial_dir.is_dir() else [ChatMessageUser(content=turns[0])]
        meta = {k: r.get(k) for k in ("quality", "component", "purpose", "steps", "tool_calls", "wall_s", "reread_tokens",
                                       "tool_validity", "tool_invalid", "tool_narrated", "tool_rejected", "tool_guarded",
                                       "compactions", "timed_out", "runtime_failed", "binary_sha") if k in r}
        samples.append(EvalSample(
            id=r["task"], epoch=int(r.get("trial", 1)), input=turns[0], target="pass",
            messages=msgs, metadata=meta,
            scores={"grader": EvalSampleScore(value="C" if r.get("pass") else "I", answer=str(r.get("notes", ""))[:500],
                                               explanation=str(r.get("notes", "")))},
            total_time=float(r.get("wall_s") or 0) or None,
        ))
    passed = sum(1 for r in rows if r.get("pass"))
    validity = [r["tool_validity"] for r in rows if isinstance(r.get("tool_validity"), (int, float))]
    metrics = {"accuracy": EvalMetric(name="accuracy", value=passed / len(rows))}
    if validity:
        metrics["tool_call_validity"] = EvalMetric(name="tool_call_validity", value=sum(validity) / len(validity))
    created = datetime.strptime(run.name[:13], "%Y%m%d-%H%M").replace(tzinfo=timezone.utc).isoformat() if run.name[:13].replace("-", "").isdigit() else datetime.now(timezone.utc).isoformat()
    log = EvalLog(
        status="success",
        eval=EvalSpec(created=created, task=f"localcode/{label}", task_id=run.name, dataset=EvalDataset(name="localcode-tasks", samples=len(rows)),
                      model=f"localcode/{model}", config=EvalConfig(epochs=max(int(r.get("trial", 1)) for r in rows)),
                      metadata={"run": run.name, "label": label}),

        results=EvalResults(total_samples=len(rows), completed_samples=len(rows),
                            scores=[EvalScore(name="grader", scorer="hidden tests/test.sh", metrics=metrics)]),
        stats=EvalStats(started_at=created, completed_at=created),
        samples=samples,
    )
    out = out_dir / f"{run.name}.eval"
    write_eval_log(log, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=[str(HERE / "runs")])
    ap.add_argument("--out", default=str(HERE / "runs" / "inspect"))
    ap.add_argument("--tasks", default=str(HERE / "tasks"))
    ap.add_argument("--view", action="store_true", help="launch `inspect view` on the exported logs")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in a.runs:
        for run in sorted(p for p in Path(d).iterdir() if p.is_dir() and (p / "results.jsonl").is_file()):
            if export_run(run, out_dir, Path(a.tasks)):
                n += 1
    print(f"{n} runs -> {out_dir}")
    if a.view:
        subprocess.run([sys.executable, "-m", "inspect_ai", "view", "--log-dir", str(out_dir)], check=False)


if __name__ == "__main__":
    main()
