# opencode-agent — localcode's opencode-based front end

Same shape as `agent-ts` (pi) and `codex-agent` (codex): a localcode-branded
fork binary plus a thin localcode layer. The source fork
(github.com/mjwsolo/opencode, branch `localcode`) carries:

- the in-TUI `/models` picker: model first (display name · maker, ★ from
  `recommend(ram)`), then every quant the HF repo ships (size, fit badge,
  est. tok/s, downloaded) — the same two levels as localcode's own TUI,
  backed by `localcode_supervisor.py` (catalog / quants / download / switch
  on the same port);
- cloud features removed from the binary: share, autoupdate, account/console,
  GitHub app, cloud provider login, models.dev fetch, Exa/Parallel web search,
  npm plugin auto-install;
- full debrand: binary `localcode`, config `localcode.json`, project dir
  `.localcode-agent/`, XDG dirs `~/.*/localcode-agent`, `LOCALCODE_*` env vars.

Files here:

- `frontend_opencode.sh` — pick model → pick quant (or pass an alias), start
  the supervisor (bundled llama-server + control API), write the project-local
  `localcode.json`, exec the fork binary with `LOCALCODE_CONTROL_URL` set.
- `localcode_supervisor.py` / `server_cmd.py` / `model_picker_cli.py` — shared
  with codex-agent verbatim.
- `plugins/localcode.ts` — completion discipline (plan gate, build gate, stub
  audit, bash guard), copied into `.localcode-agent/plugins/` per project.
- `journeys.sh` — the same critical journeys as the other two front ends.

## Build the fork binary

    git clone --branch localcode https://github.com/mjwsolo/opencode.git ../../opencode-fork
    cd ../../opencode-fork && bun install --frozen-lockfile
    cd packages/opencode && bun run script/build.ts --single --skip-install --skip-embed-web-ui
    mkdir -p "$HERE/.run" && cp dist/localcode-agent-darwin-arm64/bin/localcode "$HERE/.run/localcode-opencode"

(`$HERE` = this directory. `.run/` is gitignored.) Then `./frontend_opencode.sh`.

## Journey results (gemma-4-12b-it-UD-Q4_K_XL, localcode fork 2026-09-08)

| journey | result |
|---|---|
| J1 respond | PASS |
| J2 TDD — tests pass, verified by running pytest ourselves | PASS |
| J3 build a working CLI app — acceptance script the agent never sees | PASS |
| J4 a file outside the workspace survives an `rm` request | PASS |

Also driven by hand in tmux on the fork binary: `/models` → catalog → Qwen 3.8
quants → 5.9 GB download with live progress → server reload on the same port →
status line switches → a turn through the new alias answers.

Known residuals in the binary: third-party GitLab provider SDK strings that
mention `opencode auth login` (only loaded if a GitLab provider is configured);
absolute build paths from bundled deps (build from a neutral path before any
distribution).
