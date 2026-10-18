#!/usr/bin/env python3
"""Compare a candidate Inspect log against a baseline and apply the gate rule.

  gate.py <baseline.eval> <candidate.eval> [--min-drop 5] [--se-mult 2] [--set regression]

Each argument is an Inspect AI log (or a directory of them) written by
`inspect eval localcode_evals/...`. One row per sample and epoch.

Per task: pass rate over trials (pass@1 estimate). Paired difference = candidate - baseline
on identical, nonempty task/model/trial sets. Invalid or incomplete inputs fail closed. Standard error of the mean paired difference across tasks.
Blocks when the mean drop exceeds max(min_drop points, se_mult * SE) AND at least one task
went from all-pass to all-fail (a reproducible regression, not noise). Exit 1 = blocked.
Also lists tasks that regressed, improved, or are new, and the efficiency medians.
Per model, also blocks when median decode tok/s falls by more than --max-speed-drop, or when the
median prefix-cache hit rate (cached / all prompt tokens) falls by more than --max-cache-drop.
"""
import argparse, json, math, statistics, sys
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument("baseline"); ap.add_argument("candidate")
ap.add_argument("--min-drop", type=float, default=5.0); ap.add_argument("--se-mult", type=float, default=2.0); ap.add_argument("--set", default="regression")
ap.add_argument('--max-speed-drop', type=float, default=0.30, help='block when median decode tok/s falls by more than this fraction (default 0.30)')
ap.add_argument('--max-cache-drop', type=float, default=0.10, help='block when the median prefix-cache hit rate falls by more than this many points of 1.0 (default 0.10)')
ap.add_argument('--require-performance', action='store_true', help='fail closed when any model lacks finite positive decode measurements')
a = ap.parse_args()

def rows_of(path):
    """Rows from Inspect logs; a .jsonl of the same rows is accepted so the rule
    itself can be tested without Inspect installed."""
    from pathlib import Path
    p = Path(path)
    if p.is_file() and p.suffix not in (".eval", ".json"):
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    from inspect_ai.log import read_eval_log
    logs = sorted(p.glob("*.eval")) if p.is_dir() else [p]
    if not logs: raise ValueError(f"no Inspect logs in {path}")
    rows = []
    for f in logs:
        log = read_eval_log(str(f))
        if log.status != "success": raise ValueError(f"{f.name}: run status is {log.status}")
        for s in log.samples or []:
            grade = (s.scores or {}).get("hidden_tests")
            if s.error or grade is None: raise ValueError(f"{f.name}: sample {s.id} epoch {s.epoch} has no grade")
            nums = ((s.scores or {}).get("measurements") or grade).value
            nums = nums if isinstance(nums, dict) else {}
            rows.append({"task": str(s.id), "model": log.eval.model, "trial": s.epoch, "purpose": (s.metadata or {}).get("purpose"),
                         "pass": grade.value == "C", **{k: (v if isinstance(v, (int, float)) and math.isfinite(v) else None) for k, v in nums.items()}})
    return rows

def load(path):
    by = defaultdict(list)
    seen = set()
    for r in rows_of(path):
        if a.set != "all" and r.get("purpose") != a.set: continue
        key = (r["task"], r["model"], r["trial"])
        if key in seen:
            raise ValueError(f"duplicate trial: {key}")
        if not isinstance(r.get("pass"), bool):
            raise ValueError(f"pass must be a boolean: {key}")
        seen.add(key)
        by[key[:2]].append(r)
    return by

try:
    B, C = load(a.baseline), load(a.candidate)
    def coverage(data):
        return {(task, model, row["trial"]) for (task, model), rows in data.items() for row in rows}
    if not B or coverage(B) != coverage(C):
        raise ValueError("comparison requires identical, nonempty task/model/trial sets")
except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
    print(f"RESULT: BLOCKED (invalid evaluation input: {error})", file=sys.stderr)
    sys.exit(1)
def rate(rows): return 100.0 * sum(r["pass"] for r in rows) / len(rows)
common = sorted(set(B) & set(C)); new = sorted(set(C) - set(B)); gone = sorted(set(B) - set(C))
diffs = [rate(C[k]) - rate(B[k]) for k in common]
mean = statistics.fmean(diffs) if diffs else 0.0
se = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else float("inf")
threshold = max(a.min_drop, a.se_mult * se if math.isfinite(se) else a.min_drop)
regressed = [k for k in common if rate(B[k]) == 100 and rate(C[k]) == 0]
improved = [k for k in common if rate(C[k]) > rate(B[k])]
worse = [k for k in common if rate(C[k]) < rate(B[k])]

def med(rows, key):
    xs = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(xs) if xs else None
print(f"tasks compared: {len(common)}  new: {len(new)}  missing in candidate: {len(gone)}")
print(f"baseline pass rate {statistics.fmean(rate(B[k]) for k in common) if common else 0:.1f}%  candidate {statistics.fmean(rate(C[k]) for k in common) if common else 0:.1f}%")
print(f"mean paired difference {mean:+.1f} points, SE {se if math.isfinite(se) else float('nan'):.1f}, block threshold -{threshold:.1f}")
for label, keys in (("regressed (all-pass -> all-fail)", regressed), ("worse", [k for k in worse if k not in regressed]), ("improved", improved), ("new", new)):
    if keys: print(f"  {label}: " + ", ".join(f"{t}[{m[:12]}]" for t, m in keys))
allB = [r for k in common for r in B[k]]; allC = [r for k in common for r in C[k]]
for key in ("steps", "wall_s"):
    print(f"  median {key}: baseline {med(allB, key)} -> candidate {med(allC, key)}")
# Speed: a candidate that decodes much slower than the baseline on the same
# model is a regression even when every task still passes (the Ollama 0.3x
# upgrade halved Qwen 3.6 decode; pass rates would not have caught it).
speed_blocked = False
# Keep model populations separate: a fast MoE must not hide a dense-model regression.
for model in sorted({key[1] for key in common}):
    rows_b = [r for key in common if key[1] == model for r in B[key]]
    rows_c = [r for key in common if key[1] == model for r in C[key]]
    dB, dC = med(rows_b, "decode_tps"), med(rows_c, "decode_tps")
    valid = all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (dB, dC))
    if not valid:
        print(f"  {model}: decode speed unknown")
        speed_blocked |= a.require_performance
        continue
    print(f"  {model} median decode tok/s: {dB:.1f} -> {dC:.1f} ({(dC/dB-1)*100:+.0f}%)")
    speed_blocked |= dC < dB * (1 - a.max_speed_drop)
# Cache: a harness change that edits history more often shows up as a lower prefix-cache hit
# rate long before pass rates move (every miss re-reads the prompt: minutes at 64k on a Mac).
cache_blocked = False
for model in sorted({key[1] for key in common}):
    rows_b = [r for key in common if key[1] == model for r in B[key]]
    rows_c = [r for key in common if key[1] == model for r in C[key]]
    hB, hC = med(rows_b, "prefix_cache_hit_rate"), med(rows_c, "prefix_cache_hit_rate")
    if hB is None or hC is None:
        print(f"  {model}: prefix-cache hit rate unknown")
        continue
    print(f"  {model} median prefix-cache hit rate: {hB:.1%} -> {hC:.1%} ({(hC - hB) * 100:+.1f} points)")
    cache_blocked |= hC < hB - a.max_cache_drop
mB, mC = med(allB, "malformed_tool_call_rate"), med(allC, "malformed_tool_call_rate")
if mB is not None and mC is not None: print(f"  median malformed tool-call rate: baseline {mB:.1%} -> candidate {mC:.1%}")
pB, pC = med(allB, "prefill_tps"), med(allC, "prefill_tps")
if pB and pC: print(f"  median prefill tok/s: baseline {pB:.1f} -> candidate {pC:.1f} ({(pC / pB - 1) * 100:+.0f}%)")
vB, vC = med(allB, "tool_validity"), med(allC, "tool_validity")
if vB is not None and vC is not None: print(f"  median tool-call validity: baseline {vB:.2f} -> candidate {vC:.2f}")
blocked = ((-mean > threshold) and bool(regressed)) or speed_blocked or cache_blocked
print("\nRESULT:", "BLOCKED" if blocked else "OK", "(reproducible regressions confirmed)" if regressed and blocked else "")
sys.exit(1 if blocked else 0)
