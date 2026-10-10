# Tasks

One directory per task, Harbor-style:

```
<id>/instruction.md      what the user says (multi-turn tasks: one "## Turn N" section per turn)
<id>/task.toml           quality, component, purpose, budgets, allowed_paths, null_passes
<id>/workspace/          staged into a fresh git repo per trial
<id>/tests/test.sh       hidden from the agent; run in the workspace after the agent stops;
                         writes .eval/reward.json {"pass": bool, "notes": str}
<id>/solution/solve.sh   the oracle: applied to a fresh workspace it must make test.sh pass
```

Runner contract. Before `tests/test.sh` runs, the runner writes into `$WORKSPACE/.eval/`:
`diff.txt` (changed paths vs the initial commit), `final.txt` (the agent's last message),
`commands.log` (every bash command the agent ran, one per line), `metrics.json`
(`steps`, `wall_s`, `reread_tokens` per turn, `prompt_tokens`), `ui_errors.txt`.
`test.sh` gets `WORKSPACE` and `TASK_DIR` in the environment and must not need network.

`verify_oracles.py` proves every grader twice: the untouched workspace must FAIL
(unless `null_passes = true`, for should-not-do tasks), and the oracle must PASS.
