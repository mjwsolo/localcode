#!/usr/bin/env python3
"""Compare a candidate results.jsonl against a baseline and apply the gate rule.

  gate.py <baseline.jsonl> <candidate.jsonl> [--min-drop 5] [--se-mult 2] [--set regression]

Per task: pass rate over trials (pass@1 estimate). Paired difference = candidate - baseline
on identical, nonempty task/model/trial sets. Invalid or incomplete inputs fail closed. Standard error of the mean paired difference across tasks.
Blocks when the mean drop exceeds max(min_drop points, se_mult * SE) AND at least one task
went from all-pass to all-fail (a reproducible regression, not noise). Exit 1 = blocked.
Also lists tasks that regressed, improved, or are new, and the efficiency medians.
"""
import argparse, json, math, statistics, sys
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument("baseline"); ap.add_argument("candidate")
ap.add_argument("--min-drop", type=float, default=5.0); ap.add_argument("--se-mult", type=float, default=2.0); ap.add_argument("--set", default="regression")
ap.add_argument('--max-speed-drop', type=float, default=0.30, help='block when median decode tok/s falls by more than this fraction (default 0.30)')
a = ap.parse_args()

def load(path):
    by = defaultdict(list)
    seen = set()
    for line in open(path):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
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
except (OSError, ValueError, KeyError, TypeError) as error:
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
dB, dC = med(allB, "decode_tps"), med(allC, "decode_tps")
speed_blocked = False
if dB and dC:
    print(f"  median decode tok/s: baseline {dB:.1f} -> candidate {dC:.1f} ({(dC / dB - 1) * 100:+.0f}%)")
    speed_blocked = dC < dB * (1 - a.max_speed_drop)
    if speed_blocked: print(f"  decode fell more than {a.max_speed_drop * 100:.0f}%: BLOCKED")
pB, pC = med(allB, "prefill_tps"), med(allC, "prefill_tps")
if pB and pC: print(f"  median prefill tok/s: baseline {pB:.1f} -> candidate {pC:.1f} ({(pC / pB - 1) * 100:+.0f}%)")
vB, vC = med(allB, "tool_validity"), med(allC, "tool_validity")
if vB is not None and vC is not None: print(f"  median tool-call validity: baseline {vB:.2f} -> candidate {vC:.2f}")
blocked = ((-mean > threshold) and bool(regressed)) or speed_blocked
print("\nRESULT:", "BLOCKED" if blocked else "OK", "(reproducible regressions confirmed)" if regressed and blocked else "")
sys.exit(1 if blocked else 0)
