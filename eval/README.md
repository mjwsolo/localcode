# localcode proficiency benchmark

Measures whether localcode actually works on real coding tasks, on real hardware, with real local models. Designed to validate the "ultimate local coding tool" thesis with reproducible numbers.

## Why

Office-hours flagged "worked quite well for building an app" as the desperate-specificity gap: until we can point to named sessions with model + hardware + task + wall-clock + what-broke, the wedge isn't proven. This harness fixes that.

## Structure

```
eval/
├── README.md            ← you are here
├── rubric.md            ← how tasks are scored
├── runner.py            ← runs one task against one (model, machine) pair
├── tasks/               ← standardized tasks, one per folder
│   ├── 01-fizzbuzz/         (trivial — sanity check the loop runs at all)
│   ├── 02-fix-bug/          (small — read code, find one bug, fix it)
│   ├── 03-refactor/         (medium — split one function into two, keep tests green)
│   ├── 04-add-test/         (medium — write a pytest for an existing module)
│   └── 05-build-feature/    (hard — add a new flag end-to-end with a test)
└── results/             ← gitignored output: one JSON per run
```

Each task folder has:
- `task.md` — goal in plain English, success criteria, time budget
- `setup.sh` — copies fixtures into a fresh tmpdir (idempotent, no network)
- `verify.sh` — exits 0 if the task passed, non-zero with a reason otherwise
- `fixtures/` — starter code (if any)

## Usage

```bash
# Run one task (manual mode — you drive the TUI, harness times + verifies)
python eval/runner.py --task 01-fizzbuzz --model qwen2.5-coder-32b-q4 --mode manual

# Run all tasks (manual mode)
python eval/runner.py --all --model qwen2.5-coder-32b-q4 --mode manual

# Run one task (headless mode — REQUIRES localcode to grow a non-interactive
# entrypoint; see docs/HEADLESS.md once it exists)
python eval/runner.py --task 01-fizzbuzz --model qwen2.5-coder-32b-q4 --mode headless
```

## Output

`eval/results/{model}__{machine-slug}__{task}__{timestamp}.json`

```json
{
  "task": "01-fizzbuzz",
  "model": "qwen2.5-coder-32b-q4",
  "machine": {
    "chip": "Apple M3 Max",
    "ram_gb": 36,
    "gpu": "M3 Max 30-core",
    "os": "Darwin 25.4.0"
  },
  "started_at": "2026-05-24T18:00:00Z",
  "finished_at": "2026-05-24T18:04:12Z",
  "wall_clock_s": 252,
  "tok_total": 1847,
  "tok_per_s_avg": 7.3,
  "rounds": 4,
  "verify_exit": 0,
  "verify_reason": "all assertions passed",
  "notes": "what the user observed during the run"
}
```

## Scoring

See `rubric.md`. Short version: each task is pass/fail. A model+machine pair earns 1 point per pass. A proficiency report is `passes/total` plus median wall-clock.

## Status

- [x] Harness skeleton (this commit)
- [ ] 5 task definitions (in progress)
- [ ] Manual-mode runner
- [ ] Headless mode in localcode (requires `localcode run --goal "..."` non-interactive entrypoint — separate plan)
- [ ] First real run on M3 Max, Qwen-32B-Q4
- [ ] Published `docs/benchmarks.md`
