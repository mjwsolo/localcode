# codex-agent — localcode's Codex-based front end

Same shape as `agent-ts` (the Pi distribution): a pinned upstream binary plus a
thin localcode layer, never a maintained fork of the whole tree. The source
fork (github.com/mjwsolo/codex, branch `localcode`) carries only three small
commits: branding on the exec banner and welcome screen, and silencing the
fallback-metadata warning that fires on every local model.

- `config.toml` — the localcode profile: turboquant llama-server over the
  Responses API, static env-key auth (no ChatGPT login), codex's own approvals
  on (`on-request` + `workspace-write`).
- `frontend_codex.sh` — start the bundled server, point an isolated CODEX_HOME
  at it, hand over. A user's `~/.codex` is never read or written.
- `journeys.sh` — the critical user journeys, end to end, against a real model.

## Journey results (gemma-4-12b-it-UD-Q4_K_XL, stock codex 0.151.0)

| journey | result |
|---|---|
| J1 respond | PASS |
| J2 TDD — tests pass, verified by running pytest ourselves | PASS |
| J3 build a working CLI app — acceptance script the agent never sees | PASS |
| J4 sandbox — a file outside the workspace survives `rm` | PASS |
| J5 no fallback-metadata warning | FAIL on stock; **verified fixed on the fork binary** |

## Fork binary (mjwsolo/codex, branch `localcode`)

Built with cargo 1.95 and verified live against gemma-4-12b: banner reads
`localcode (codex core)`, reply returned, zero metadata-warning lines. Use it
via `LOCALCODE_CODEX_BIN=/path/to/target/release/codex ./frontend_codex.sh`.

Notable vs the Pi front end: codex brings its own approval system and sandbox
(J4 passed with zero localcode code), and its own web search — the two largest
extensions we had to write for Pi.
