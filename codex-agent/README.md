# codex-agent — localcode's Codex-based front end

Same shape as `agent-ts` (the Pi distribution): a pinned upstream binary plus a
thin localcode layer, never a maintained fork of the whole tree. The source
fork (github.com/mjwsolo/codex, branch `localcode`) carries only three small
commits: branding on the exec banner and welcome screen, and silencing the
fallback-metadata warning that fires on every local model.

- `localcode_supervisor.py` — owns the bundled llama-server and serves the
  in-TUI `/model` picker over a localhost control port (`LOCALCODE_CONTROL_URL`):
  `GET /catalog` (every catalog model, ★ from `recommend()`), `GET /quants`
  (every quant the HF repo ships, size, fit badge, downloaded marker),
  `POST /select` (download if needed, then restart the server on the SAME
  port), `GET /status` (live progress). Not a reimplementation: it imports
  localcode's `models_catalog`, `hf_quants` and `bootstrap.download_model`.
- `model_picker_cli.py` — the same two-level picker, pre-launch, for a first run
  with no model argument.
- `config.toml` — the localcode profile: turboquant llama-server over the
  Responses API, static env-key auth (no ChatGPT login), codex's own approvals
  on (`on-request` + `workspace-write`).
- `frontend_codex.sh` — start the bundled server, point an isolated CODEX_HOME
  at it, hand over. A user's `~/.codex` is never read or written.
- `journeys.sh` — the critical user journeys, end to end, against a real model.

## Journey results (stock codex 0.151.0)

Two models. Every PASS verified by scripts the agent never sees.

| journey | result |
|---|---|
| J1 respond | PASS |
| J2 TDD — tests pass, verified by running pytest ourselves | PASS |
| J3 build a working CLI app — acceptance script the agent never sees | PASS |
| J4 sandbox — a `$HOME` file survives `rm` | PASS (both models) |
| J5 no fallback-metadata warning | FAIL on stock; **verified fixed on the fork binary** |

Qwen 3.8 27B: J1-J4 all PASS on a clean run. Two lessons from iteration:
a first Qwen run failed J3 with an empty workdir because other model loads
shared the GPU mid-run (rerun clean: PASS), and J4 originally placed its canary
in `$TMPDIR`, which `workspace-write` legitimately allows — it now uses `$HOME`,
which the sandbox really does protect.

## Fork binary (mjwsolo/codex, branch `localcode`)

Built with cargo 1.95 and verified live against gemma-4-12b: banner reads
`localcode (codex core)`, reply returned, zero metadata-warning lines. Use it
via `LOCALCODE_CODEX_BIN=/path/to/target/release/codex ./frontend_codex.sh`.

Notable vs the Pi front end: codex brings its own approval system and sandbox
(J4 passed with zero localcode code), and its own web search — the two largest
extensions we had to write for Pi.
