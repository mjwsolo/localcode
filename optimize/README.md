# optimize/ — GEPA-style reflective prompt optimizer

Offline, **dev-time** tooling that searches for a better LocalCode system
prompt. It is a separate top-level package (sibling of `src/` and `tests/`):
it MAY import from `localcode` (to read the live system prompt and reuse the
benchmark harness), but **nothing in `src/localcode` imports from here**. It
is never part of the shipped runtime.

It implements the GEPA idea: instead of mutating prompts blindly, an LLM
*reflects on the benchmark's natural-language feedback* (which tasks failed
and why) and proposes a targeted rewrite, while a Pareto frontier of
candidates is kept across iterations.

## How the loop works

1. **Candidate** = a system-prompt variant. The seed is the live
   `SYSTEM_PROMPT` (read from `localcode.agent.prompts`).
2. **Evaluate** (`evaluator.py`): runs the *existing* bench harness
   (`scripts/bench_prompt_variants.py` + `bench_prompt_hard.py` task set,
   extractor, and hidden-assert scorer) and returns **both** a scalar
   pass-rate **and** textual feedback — the failing tasks, the generated
   code, and the assertion/exception that sank each one.
3. **Reflect + propose** (`reflector.py`): an LLM reads the candidate + its
   feedback and emits an improved variant (fenced, with the runtime
   placeholders preserved or we fall back to the parent).
4. **Frontier + selection** (`frontier.py`): a Pareto frontier keeps
   non-dominated candidates (e.g. code-gen vs agentic trade-offs); parents to
   mutate are sampled from the frontier, not just the single best.
5. **Loop** (`loop.py`): repeats for N iterations or until `patience`
   iterations pass with no improvement. The global best is tracked, so the
   result never regresses even if an iteration proposes a worse prompt.
6. Output: best prompt + score history written to a JSON file.

Why this beats random search: proposals are driven by the textual feedback
(directed, not random), and a frontier preserves diverse trade-off regions
instead of collapsing onto one local optimum.

## Run it (CLI)

```bash
# from the repo root
python -m optimize.gepa --model gemma-4-12b-it-UD-Q4_K_XL.gguf --iterations 6
# optional: use a STRONGER model to propose rewrites than the one being optimized
python -m optimize.gepa --model <small.gguf> --reflector-model <strong.gguf> --iterations 8
```

Useful flags: `--proposals-per-iter`, `--patience`, `--k` (bench samples per
task), `--seed`, `--out`. Requires a llama-server-runnable GGUF already
downloaded (the harness spins the server up/down for you).

## Tests

```bash
uv run --with pytest python -m pytest optimize/tests -q
```

The tests cover the **pure** parts with **no model and no network**: the
candidate dataclass, the Pareto/selection logic (dominance, ties,
no-regression), the feedback renderer, the reflection meta-prompt builder and
fenced-output extractor, and the full loop wired with a **mocked evaluator and
mocked reflector** (asserting it improves over iterations, keeps the best on
regressions, handles ties, and early-stops on patience).

## Limitations (be honest)

- **Only as good as the bench harness.** The optimizer maximizes exactly what
  `bench_prompt_*` measures (a small set of Python code-gen tasks). It can
  overfit to those tasks; broaden the task set for a real run.
- **Model-specific.** A prompt tuned for one model/quant won't necessarily
  transfer to another. Re-run per target model.
- **The reflector should be strong.** Prompt-design reasoning is hard; a weak
  local model makes weak proposals. Point `--reflector-model` at the best
  model you can run.
- **PoC scope.** Hand-rolled reflective loop (no DSPy, no extra deps).
  Single-server-per-role, modest sampling — tune `--k`/`--iterations` for
  cost vs signal. `exec`-based scoring runs generated code; run in a sandbox
  if that matters to you.
