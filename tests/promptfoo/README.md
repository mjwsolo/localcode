# LocalCode prompt evaluation with `promptfoo`

Statistical A/B testing of our system prompt against the real
`llama-server`. Answers: "if I remove rule N, does pass rate on the
scenario suite drop?"

## Quick start

```bash
# from the repo root
cd tests/promptfoo

# one-off eval of the current prompt (assumes llama-server on :8081)
npx promptfoo@latest eval

# open the local dashboard (diffs, per-scenario drill-down)
npx promptfoo@latest view

# A/B: test two versions of the prompt
#   1. edit SYSTEM_PROMPT in src/localcode/agent.py (try removing a rule)
#   2. commit
#   3. run `npx promptfoo eval` → results saved under ~/.promptfoo
#   4. checkout the old commit, run again
#   5. `npx promptfoo view` diffs the two runs side-by-side
```

## What this tests

Each scenario is a (system prompt, user message, structural
assertions) triple. Promptfoo runs each scenario **N times** (config:
`defaultTest.options.repeat`) because IQ2/IQ3-quantized models are
stochastic — a single run is noise. Aggregate pass rate with confidence
intervals is what we actually care about.

Current v1 scenarios (see `promptfooconfig.yaml`):

1. `short-chat` — does the model respond at all + not refuse
2. `narration` — multi-step task must have per-step prose (catches the
   "silent tool burst + final essay" regression from session 2026-04-24)
3. `ambiguous-build` — model must ASK a clarifying question for an
   underspecified UI-app request (rule 22), not dive into tool calls

## Interpreting results

Promptfoo prints a pass/fail grid per scenario × provider × repeat.
Pay attention to:

- **Pass rate**: `7/9` on a scenario means 7 of 9 runs passed.
  Anything ≥ 80% is usually fine; < 50% means the prompt isn't
  steering reliably.
- **Variance across repeats**: if the same scenario passes 3/3 on
  one run and 0/3 on another, the prompt is on a knife-edge —
  usually means a rule is weakly worded or conflicts with another.
- **Time per run**: llama-server is slow (~27 tok/s decode on M4);
  2–5 seconds per short scenario is normal.

## The feedback loop we actually want

Today our "prompt engineering" process is: change a rule → eyeball
one session → ship. Variance means one session can't tell us whether
a change helped. The promptfoo loop:

1. Define the scenarios we care about (real user tasks that have
   broken in past sessions).
2. Every time someone proposes a prompt change, run the suite
   before + after with `repeat: 10`.
3. Ship the change only if the aggregate pass rate holds or improves
   within confidence bounds.

This is exactly what top labs do internally (N=5–20 reps, pass@k with
CIs, ablation diffs). We can run it locally with no cloud calls.

## Not in v1

- **LLM-as-judge** (`llm-rubric` assertion type) — requires a
  stronger model than our local one to grade outputs. Can add
  optionally via `OPENAI_API_KEY` env var when someone wants to
  measure quality beyond structural pass/fail.
- **Reasoning-mode variant** — add a second prompt file that sets
  `reasoning_rules` non-empty and point a scenario at it.
- **Regression against previous commits** — promptfoo stores results
  under `~/.promptfoo`; we'll add a script that runs against the last
  N tagged commits to track prompt-behavior drift over time.

## Why NOT just extend our existing harness

Our `tests/e2e/` harness already runs scenarios against the real agent
loop. It catches infrastructure regressions (server memory, tool
dispatch, recovery paths) that promptfoo can't because promptfoo
talks directly to llama-server, not through our agent.

The two are complementary:
- **`tests/e2e/`** — end-to-end integration, catches plumbing bugs
- **`tests/promptfoo/`** — prompt-level A/B, catches rule-change
  regressions statistically

Run promptfoo when changing the system prompt. Run e2e when changing
agent/tool/server code.
