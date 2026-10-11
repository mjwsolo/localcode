# localcode evals

The task suite runs on [Inspect AI](https://inspect.aisi.org.uk), the UK AI Security
Institute's evaluation framework. Inspect owns the run: datasets, epochs, limits, logs,
scoring, comparison and the viewer. This directory adds only what is specific to
LocalCode: the tasks, a solver that drives the real UI, and a grader per task.

Nothing here is part of the `localcode` package, and nothing leaves the machine.

## Run it

```
pip install -e evals                      # inspect-ai plus this suite, in any environment
inspect eval localcode_evals/regression --model localcode/<model>
inspect view
```

`<model>` is an installed GGUF name without `.gguf`, or a catalog group key such as
`qwen-3.8-27b`, in which case the picker chooses the quant exactly as it would for a
user. Models are never downloaded for you.

| Want | Command |
| --- | --- |
| The release-blocking set | `inspect eval localcode_evals/regression --model localcode/<model>` |
| The harder, informational set | `inspect eval localcode_evals/tracking --model localcode/<model>` |
| Named tasks | `inspect eval localcode_evals/localcode -T ids=fix-one-bug,add-test --model ...` |
| Repeat each task (pass@k, pass^k) | `--epochs 3 --epochs-reducer mean,pass_at_3,at_least_3` |
| A 16 GB Mac's choices | `-T ram_gb=16 --model localcode/<group key>` |
| Several models, resumable | `inspect eval-set localcode_evals/regression --model localcode/a,localcode/b --log-dir logs/x` |
| A/B a launcher setting | `-T env='{"LOCALCODE_LLAMA_CPP_MTP_DRAFT": "0"}'`, then compare the two logs |
| Compare two runs, apply the gate rule | `python3 evals/gate.py <baseline log or dir> <candidate log or dir>` |
| Keep the workspace of a failing sample | `--no-sandbox-cleanup` |

The checkout under test is the one this directory sits in; `LOCALCODE_EVALS_REPO`
points at another. The supervisor runs under `LOCALCODE_PY`, else `<checkout>/.venv/bin/python`.

## HarnessBench, run locally

`inspect eval localcode_evals/harnessbench --model localcode/<model>` runs
[HarnessBench](https://github.com/Qihoo360/harness-bench) through the same solver: office,
data, SRE, knowledge and codebase jobs in a real workspace, each graded by the task's own
programmatic oracle. It measures the agent harness, which is the product. Benchmarks that
mostly rank the model (Aider Polyglot, LiveCodeBench, SWE-bench) are deliberately absent.

No Docker, no cloud. The data is not in this repository and is never fetched for you:
`python -m localcode_evals.data` shows whether it is present, its size and the command.

It takes `-T ids=`, `-T classes=` (a word in the task class, e.g. `classes=Software,SRE`),
`-T limit=`, `-T ram_gb=` and `-T env=`, and reports `solved` (every check passes) and
`outcome_score` (the oracle's weighted fraction).

Left out rather than scored wrong: tasks that need a public tunnel or per-round hooks.
The benchmark's optional LLM quality rubric is not used, so scores are deterministic.

## What a sample is

Each directory under `tasks/` is one sample.

| File | Role |
| --- | --- |
| `instruction.md` | What the user typed. `## Turn N` sections make a multi-turn session. |
| `workspace/` | Copied into the sample's sandbox directory and committed as a fresh git repo. |
| `task.toml` | Quality, component, purpose, time budget, optional MCP servers and setup command. |
| `tests/test.sh` | The hidden grader. It writes `.eval/reward.json`. |
| `solution/solve.sh` | The oracle. `verify_oracles.py` proves each grader fails an untouched workspace and passes the oracle. |

`tasks/README.md` documents the `.eval/` files a grader may read.

## What runs

`localcode_ui()` (in `localcode_evals/agent.py`) starts one isolated supervisor and
llama-server per model (throwaway home, spare ports, one slot, only the GGUF files
shared) and, per sample, runs `localcode-ui run --format json` once per turn in the
sandbox directory. The UI talks to llama-server directly, as it does for a user. Inspect
is deliberately not placed between them, because prompt caching, slots and streaming are
part of what is measured.

What the UI did becomes the sample transcript: every assistant message, tool call and
tool result. The server's token counts are recorded as the sample's model usage.

Two scorers run on every sample:

- `hidden_tests`: the task's grader. Correct or incorrect, with the grader's note as the
  explanation and the diff in the score metadata. A timed-out run or a UI that exited
  non-zero cannot pass.
- `measurements`: `tool_validity` (of the calls the model meant to make, how many
  executed), `steps`, `tool_calls`, `wall_s`, `reread_tokens_max` (tokens re-read after
  the first turn), `prompt_tokens_max`, and the server's own `decode_tps` and
  `prefill_tps`. Memory, swap, battery and per-request timings are in the metadata.

## What is covered, and by what

Every task is tagged with a quality (capability, reliability, safety_scope, efficiency,
fidelity), a component (model, harness, runtime, ui) and a purpose (regression,
tracking). Regression tasks are expected near 100%. Tracking tasks show where to work next.

| Dimension | Unit + Checks (no model, CI) | Gate (model loads and acts once) | Inspect tasks (model does work) |
| --- | --- | --- | --- |
| Correctness of work | - | tool loop, runtime turn | capability tasks |
| Tool calls execute | tool schemas identical | tool loop | `tool_validity` |
| Prompt cache and cost | constant prompt, budgets | - | `stays-cached`, `reread_tokens_max` |
| Long sessions and compaction | forced-compaction checklist | - | `ten-turn-session`, `compaction-depth` |
| Safety and scope | path guards, bash guard | - | safety_scope tasks, `injection-in-file` |
| Speed | - | drafter accepted drafts | `decode_tps`, `prefill_tps`; `gate.py` blocks a >30% decode drop |
| Memory, heat, battery | - | - | peak RSS, swap growth, CPU speed limit, battery drop |
| Vision | - | image request answered | - |
| Every RAM tier | fit tests | - | `-T ram_gb=<n>` |
| Installed package, not checkout | CI field job | release install matrix | - |

Known gaps: real 16 and 24 GB hardware (`ram_gb` emulates the picker's choices, not
memory pressure), vision inside a task, and the browser tasks, which need a browser build
already installed for Playwright.

## The other pieces

- `checks/`: pytest, no model. Canned servers drive the real UI and assert the plumbing:
  identical system prompt across a session, tool schemas, budgets, no runtime errors.
  `python -m pytest evals/checks -q`. Runs on every PR.
- `verify_oracles.py`: proves every grader both ways. Runs on every PR.
- `gate.py`: reads two Inspect logs. Requires identical, nonempty task, model and epoch
  coverage, then blocks only when the mean paired drop exceeds max(5 points, 2 SE) and at
  least one task went from all-pass to all-fail, or median decode speed fell by more than
  30%.
- `nightly.sh <checkout> [model ...]`: Checks, oracles, `regression` at 3 epochs per
  model, and the gate against that model's previous completed run. Refuses to run while
  an interactive localcode session holds the supervisor port.
- `probe_server.py`: a direct server probe used by the speed tests.

## Reading results honestly

One epoch is one observation. A single failure on a 16-sample run moves the score by six
points, so compare runs with `gate.py` or with several epochs, not by eye. RSS covers
the supervisor's process tree and swap growth is system-wide, so background apps affect
both. Time to first token is not measured; prefill time is not relabelled as it.
A changed setting is only proven by the `runtime` block in the sample metadata, which
records the context, slots and overrides the server actually ran with.
