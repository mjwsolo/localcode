#!/usr/bin/env python3
"""Compare a candidate results.jsonl against a baseline and apply the gate rule.

  gate.py <baseline.jsonl> <candidate.jsonl> [--min-drop 5] [--se-mult 2] [--set regression]

Per task: pass rate over trials (pass@1 estimate). Paired difference = candidate - baseline
on tasks present in both. Standard error of the mean paired difference across tasks.
Blocks when the mean drop exceeds max(min_drop points, se_mult * SE) AND at least one task
went from all-pass to all-fail (a reproducible regression, not noise). Exit 1 = blocked.
Also lists tasks that regressed, improved, or are new, and the efficiency medians.
"""
import argparse, json, math, statistics, sys
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument("baseline"); ap.add_argument("candidate")
ap.add_argument("--min-drop", type=float, default=5.0); ap.add_argument("--se-mult", type=float, default=2.0); ap.add_argument("--set", default="regression")
a = ap.parse_args()

def load(path):
    by = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if a.set != "all" and r.get("purpose") != a.set: continue
        by[(r["task"], r["model"])].append(r)
    return by

B, C = load(a.baseline), load(a.candidate)
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
blocked = (-mean > threshold) and bool(regressed)
print("\nRESULT:", "BLOCKED" if blocked else "OK", "(reproducible regressions confirmed)" if regressed and blocked else "")
sys.exit(1 if blocked else 0)
