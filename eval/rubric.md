# Scoring rubric

## Task pass criteria

Each task defines its own `verify.sh`. Exit 0 = pass. Non-zero = fail with reason.

Verification is mechanical (assertion-style), not subjective. No "looks good" grading. If `verify.sh` passes, the task passes. Period.

## Per-run metrics captured

- **wall_clock_s** — time from "user submits goal" to "agent ends turn with no tool calls" (or hard timeout)
- **rounds** — how many model→tool→model cycles the loop took
- **tok_total** — total tokens generated (prompt + completion across all turns)
- **tok_per_s_avg** — token throughput, the headline local-model metric
- **verify_exit** — 0 if task passed, non-zero otherwise
- **stalls** — count of times the agent had to be nudged by recovery (`detect_stall`)
- **bailouts** — count of MAX_ROUNDS hits

## Time budgets per task

| Task | Soft budget | Hard timeout | Why |
|---|---|---|---|
| 01-fizzbuzz | 2 min | 5 min | Sanity check; should not be hard |
| 02-fix-bug | 5 min | 15 min | One file, one bug |
| 03-refactor | 10 min | 25 min | Edit + keep tests passing |
| 04-add-test | 10 min | 25 min | Reading comprehension + pytest fluency |
| 05-build-feature | 20 min | 45 min | Multi-file, end-to-end |

Wall clock over hard timeout = automatic fail.

## Proficiency report format

A run of all 5 tasks against a (model, machine) pair produces:

```
PROFICIENCY REPORT — Qwen-32B-Q4 on M3 Max (36GB)
==================================================
Tasks passed: 4/5
Median wall-clock: 312s
Average tok/s: 7.1
Total stalls: 2 (both in 03-refactor)
Total bailouts: 0

  01-fizzbuzz       PASS   62s    9.1 tok/s   3 rounds
  02-fix-bug        PASS   180s   7.4 tok/s   5 rounds
  03-refactor       PASS   480s   6.8 tok/s   12 rounds   2 stalls
  04-add-test       PASS   320s   7.2 tok/s   8 rounds
  05-build-feature  FAIL   1530s  6.5 tok/s   24 rounds   bailout — MAX_ROUNDS

Verdict: GOOD for tasks up to medium complexity. Hard tasks bail out.
```

## What "ultimate local coding tool" means quantitatively

Working definition for v1.0:
- **4/5 tasks pass** on a recommended-spec machine (M3 Max 36GB or equivalent)
- **Median wall-clock < 5 min** across the first 4 tasks
- **Tok/s > 5** on the 32B-Q4 reference model
- **0 manual interventions** needed during a task (no "I had to nudge it")

Anything less and we haven't earned the word "ultimate" yet.
