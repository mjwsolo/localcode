# Performance and quality validation

`/server` shows recent prompt-processing and generation speeds separately, the
active context budget, and estimated prompt reuse. It warns when multiple
llama-server processes may be competing for GPU or memory. Missing measurements
remain unknown. The reuse figure is an estimate from server timing/release logs,
not a guarantee that every prefix is cached. CPU power reports do not establish
that the GPU is unthrottled.

## Context and verification

LocalCode prunes old tool output from each request. For local models the
protected tool window is 25% of the usable input budget (1,024–40,000 estimated
tokens); pruning needs at least 12.5% reclaimable (512–20,000 tokens). The last
two user turns, skills, unsuccessful and unfinished tools remain protected.
Full old outputs remain in session storage. Pruning is decided when a new turn
starts and changes the prefix once per prune; it is useful only when the saved
future processing outweighs that reread.

Tool previews are limited to 400 lines and 4–16 KiB depending on the initial
context size. The native tool implementation saves full truncated output and
returns its path. Compaction is asked to preserve constraints, decisions,
actual check results and unfinished work. `LOCALCODE_COMPACTION_CHECKLIST=0`
disables that extra checklist.

After edits with no observed check, the completion hook asks once for the
smallest relevant existing check, a focused repair if needed, and an honest
report if checking is unavailable. This runs through normal tools and their
permissions; it does not execute repository scripts itself. Existing loop and
repeated-failure guards still apply. Set `LOCALCODE_TARGETED_VERIFY=0` to disable
this extra prompt. The older explicitly opted-in automatic-check mode is
unchanged.

An actual provider `length` finish triggers at most one continuation per user
turn. Interruption, Plan mode and a stopped task prevent it. A second limit is surfaced as a warning. This does not
increase the per-request output ceiling, nor guarantee arbitrary output length.
The model's own guesses about its token count never trigger continuation.

## Measure before tuning

1. Close unused model servers yourself. Run on consistent power and record the
   model, quantization, chip, RAM, server version and context settings.
2. Probe the server directly with synthetic prompts:

   ```sh
   python evals/probe_server.py --port PORT --auth-file PATH_TO_RUN/auth.json
   ```

   The probe reports first streamed content/reasoning/tool token latency,
   total time, server timings and usage. Repeated requests test prefix reuse.
   It fails on incomplete streams. Credentials and response text are not printed.
3. Run `inspect eval localcode_evals/regression --model localcode/<model>` (see `evals/README.md`) on the same model and fixed coding tasks. The patched
   JSON client emits one content-free `first_text_delta` timestamp. Results include
   `end_to_end_first_text_s` per turn, including harness startup, any preceding tool work and transport;
   this is distinct from server-only prefill timings. Older clients leave it
   unknown. Compare correctness as well as latency.
4. Pass `-T env='{"LOCALCODE_OVERRIDE_BATCH": "512"}'` and the like for sequential context, batch, KV and drafting experiments.
   Preserve a baseline and reverse order between repetitions. A RAM override
   exercises policy; it does not simulate a real lower-memory GPU.
5. `evals/gate.py --require-performance` rejects missing decode measurements and
   checks speed separately for each model. A fast model cannot hide a slowdown
   in another. The default gate is a screening check, not statistical proof of
   quality equivalence. Review every task regression before promotion.

Stored launch recommendations are scoped to exact chip/RAM identity and validated
before application. Older entries without hardware identity are ignored; re-run
tuning on the current machine to create a new entry. Explicit environment
settings continue to override recommendations. No universal faster settings are
claimed for an unmeasured model or Mac.

Keep raw benchmark runs under ignored `evals/runs/` or outside the checkout.
Never publish user prompts, auth files, models or session histories with a PR.
