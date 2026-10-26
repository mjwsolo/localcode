#!/usr/bin/env python3
"""Helper for scripts/release_eval.sh: read two Harness-Bench runs and compare them.

    release_eval.py summarize <inspect log or directory> [--out run.json]
    release_eval.py compare <previous run> <candidate run> [--score-margin 0.05] [--max-slowdown 0.25]
    release_eval.py previous --candidate 0.5.41rc1 [--index-json file]
    release_eval.py tasks

A "run" is an Inspect AI log (or a directory of them) or the JSON that
`summarize` writes. `compare` exits 0 when the candidate is acceptable, 1 when
its mean score fell by more than the margin or it took more than the allowed
fraction longer, and 2 when the runs cannot be compared (different or missing
tasks). Reading Inspect logs imports inspect_ai lazily; nothing else needs it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

SCORE = "harnessbench_outcome"
DEFAULT_TASKS = ("002-monorepo-affected-build", "010-layered-config-resolution", "035-workspace-dependency-resolution",
                 "092-bounded-cache-eviction-repair", "089-file-watcher-atomic-save-repair", "120-http-cookie-jar-policy-repair")


class Incomparable(Exception):
    pass


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def summarize_log(path: str | Path) -> dict:
    """{"tasks": {id: {"score", "passed", "wall_s", "error"}}} from Inspect logs."""
    from inspect_ai.log import read_eval_log
    p = Path(path)
    logs = sorted(p.glob("*.eval")) if p.is_dir() else [p]
    if not logs:
        raise Incomparable(f"no Inspect logs in {path}")
    tasks: dict[str, dict] = {}
    for f in logs:
        log = read_eval_log(str(f))
        for s in log.samples or []:
            scores = s.scores or {}
            outcome = scores.get(SCORE)
            value = outcome.value if outcome is not None and isinstance(outcome.value, dict) else {}
            measured = scores.get("measurements")
            wall = _num((measured.value if measured is not None and isinstance(measured.value, dict) else {}).get("wall_s"))
            tasks[str(s.id)] = {"score": _num(value.get("outcome_score")) or 0.0, "passed": bool(value.get("task_success")),
                                "wall_s": wall, "error": bool(s.error) or outcome is None}
    return {"tasks": tasks}


def load_run(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir() and (p / "run.json").is_file():  # a summary kept beside the logs (also what the tests use)
        p = p / "run.json"
    if p.is_file() and p.suffix == ".json":
        data = json.loads(p.read_text())
        if "tasks" not in data:
            raise Incomparable(f"{path} is not a run summary")
        return data
    return summarize_log(p)


def totals(run: dict) -> dict:
    rows = run["tasks"]
    n = len(rows)
    return {"tasks": n, "mean_score": sum(r["score"] for r in rows.values()) / n if n else 0.0,
            "passed": sum(1 for r in rows.values() if r["passed"]),
            "wall_s": sum(r["wall_s"] or 0.0 for r in rows.values()),
            "errors": sum(1 for r in rows.values() if r.get("error")),
            "timed": n > 0 and all(r["wall_s"] is not None for r in rows.values())}


def compare(previous: dict, candidate: dict, score_margin: float = 0.05, max_slowdown: float = 0.25) -> dict:
    a, b = set(previous["tasks"]), set(candidate["tasks"])
    if not a or not b:
        raise Incomparable("a run has no tasks")
    if a != b:
        raise Incomparable(f"different tasks: only previous {sorted(a - b)}, only candidate {sorted(b - a)}")
    pt, ct = totals(previous), totals(candidate)
    reasons = []
    drop = pt["mean_score"] - ct["mean_score"]
    if drop > score_margin + 1e-12:
        reasons.append(f"mean score fell {drop:.3f} (limit {score_margin:.3f})")
    slowdown = None
    if pt["timed"] and ct["timed"] and pt["wall_s"] > 0:
        slowdown = ct["wall_s"] / pt["wall_s"] - 1
        if slowdown > max_slowdown + 1e-12:
            reasons.append(f"{slowdown * 100:.0f}% slower (limit {max_slowdown * 100:.0f}%)")
    elif not (pt["timed"] and ct["timed"]):
        reasons.append("wall time missing for some tasks, cannot judge speed")
    return {"previous": pt, "candidate": ct, "score_drop": drop, "slowdown": slowdown, "ok": not reasons, "reasons": reasons,
            "regressed": sorted(t for t in a if previous["tasks"][t]["passed"] and not candidate["tasks"][t]["passed"]),
            "fixed": sorted(t for t in a if candidate["tasks"][t]["passed"] and not previous["tasks"][t]["passed"])}


def render(result: dict, previous_label: str = "previous", candidate_label: str = "candidate") -> str:
    def row(label, t):
        return f"{label:<11} score {t['mean_score']:.3f}   passed {t['passed']}/{t['tasks']}   wall {t['wall_s']:.0f}s   errors {t['errors']}"
    change = f"score change {-result['score_drop']:+.3f}"
    if result["slowdown"] is not None:
        change += f"   time change {result['slowdown'] * 100:+.0f}%"
    lines = [row(previous_label, result["previous"]), row(candidate_label, result["candidate"]), change]
    if result["regressed"]:
        lines.append("passed before, fails now: " + ", ".join(result["regressed"]))
    if result["fixed"]:
        lines.append("failed before, passes now: " + ", ".join(result["fixed"]))
    lines.append("RESULT: ok" if result["ok"] else "RESULT: BLOCKED - " + "; ".join(result["reasons"]))
    return "\n".join(lines)


def _final(v: str):
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", v)
    return tuple(map(int, m.groups())) if m else None


def previous_release(candidate: str, versions) -> str | None:
    """Newest final release strictly older than the candidate's base version."""
    m = re.match(r"\d+\.\d+\.\d+", candidate)
    if not m:
        raise ValueError(f"not a version: {candidate}")
    base = _final(m.group(0))
    older = [(t, v) for v in versions if (t := _final(v)) and t < base]
    return max(older)[1] if older else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("run")
    s.add_argument("--out")
    c = sub.add_parser("compare")
    c.add_argument("previous")
    c.add_argument("candidate")
    c.add_argument("--score-margin", type=float, default=0.05)
    c.add_argument("--max-slowdown", type=float, default=0.25)
    c.add_argument("--previous-label", default="previous")
    c.add_argument("--candidate-label", default="candidate")
    c.add_argument("--json-out")
    v = sub.add_parser("previous")
    v.add_argument("--candidate", required=True)
    v.add_argument("--index-json", help="PyPI JSON file to read instead of fetching")
    sub.add_parser("tasks")
    a = p.parse_args(argv)
    if a.cmd == "tasks":
        print(",".join(DEFAULT_TASKS))
        return 0
    if a.cmd == "previous":
        if a.index_json:
            data = json.loads(Path(a.index_json).read_text())
        else:
            with urllib.request.urlopen("https://pypi.org/pypi/localcode/json", timeout=30) as r:
                data = json.load(r)
        found = previous_release(a.candidate, list(data["releases"]))
        if not found:
            print("no earlier final release found", file=sys.stderr)
            return 2
        print(found)
        return 0
    try:
        if a.cmd == "summarize":
            text = json.dumps(load_run(a.run), indent=2)
            if a.out:
                Path(a.out).write_text(text + "\n")
            else:
                print(text)
            return 0
        result = compare(load_run(a.previous), load_run(a.candidate), a.score_margin, a.max_slowdown)
    except Incomparable as exc:
        print(f"cannot compare: {exc}", file=sys.stderr)
        return 2
    print(render(result, a.previous_label, a.candidate_label))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
