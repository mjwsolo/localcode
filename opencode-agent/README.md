# opencode-agent — localcode's OpenCode-based front end

Same shape as `agent-ts` (Pi) and `codex-agent` (Codex): a pinned upstream
binary plus a thin localcode layer. The source fork
(github.com/mjwsolo/opencode, branch `localcode`) carries the profile.

- `opencode.json` — turboquant llama-server via `@ai-sdk/openai-compatible`
  (plain chat/completions), `share` disabled, `autoupdate` off, friendly model
  names. Ships project-local so `~/.config/opencode` is never touched.
- `frontend_opencode.sh` — start the bundled server, write the project-local
  config, hand over.
- `journeys.sh` — the same critical journeys as the other two front ends.

## Journey results (gemma-4-12b-it-UD-Q4_K_XL, opencode 1.18.27)

| journey | result |
|---|---|
| J1 respond | PASS |
| J2 TDD — tests pass, verified by running pytest ourselves | PASS |
| J3 build a working CLI app — acceptance script the agent never sees | PASS |
| J4 a file outside the workspace survives an `rm` request | PASS |

4/4 — the only front end to pass every journey on the stock binary.
