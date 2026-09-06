# Handover: localcode agent-frontend work (for Fable 5.1)

Reference machine: Apple Silicon, 128 GB unified memory.
Read `~/.claude/.../memory/MEMORY.md` first — especially
`frontend-exploration-branches`, `thinking-off-default-and-cap-recovery`,
`optimize-for-1m-users-generally`, `no-claude-anthropic-attribution`.

## The product rule that outranks everything

localcode = `pip install localcode` → run → **pick model first, then quant** →
download with progress → code. Local inference; prompts/files stay local;
network is fine for downloads/web tools but NOTHING phones home invisibly.
Model picker semantics (non-negotiable, from `src/localcode/tui/screens/model_picker.py`):
- Level 1: curated `MODEL_GROUPS` — display name · maker — EVERY catalog model,
  downloaded or not, ★ from `recommend(ram_gb)`.
- Level 2: EVERY quant the HF repo ships (`hf_quants.fetch_quants`), size in GB,
  fit badge (`fit_badge`: ✓ fits ≤55% RAM / ~ tight ≤65% / ✗ too big),
  downloaded marker. Curation is families-only; quants are never curated.
Every deviation from this has been rejected. Do not invent your own UI.

## Repo / branch map (all pushed; main + PyPI untouched — keep it that way)

| where | branch | what |
|---|---|---|
| mjwsolo/localcode | `integrate/pi-frontend` | Pi-based front end — most complete |
| mjwsolo/localcode | `integrate/codex-frontend` | Codex-based front end (`codex-agent/`) |
| mjwsolo/localcode | `integrate/opencode-frontend` | OpenCode-based (`opencode-agent/`) |
| mjwsolo/codex | `localcode` | source fork; debranded; build w/ cargo 1.95 (`~/.cargo/bin`) |
| mjwsolo/opencode | `localcode` | profile only |

Worktrees on disk: `~/Desktop/Github/localcode-{pi,codex,opencode}`.
Main checkout `~/Desktop/Github/localcode` is on `integrate/fork-bump` — leave it.
NEVER commit to main, never push tags (tags drive PyPI; live = v0.3.71).

## Verified-working today (don't re-litigate, re-verify after changes)

- Pi: wheel installs & runs (`LOCALCODE_FRONTEND=agent localcode`); first-run
  picker → real download (396MB & 7.4GB proven) → load → chat, driven via tmux;
  redaction to disk; approval gate; web_search/fetch (SSRF-guarded);
  launch_app; code_navigation; 8 contract tests (`npm test` in agent-ts).
- Codex: journeys 4/5 stock + fork fixes; fork binary branded; interactive
  turn via tmux over `/v1/responses`.
- OpenCode: journeys 4/4; interactive turn verified.
- Thinking-off verified on ALL THREE wires (this faked a 2x slowdown once):
  pi → `before_provider_request` chat_template_kwargs; codex →
  `model_reasoning_effort="none"` (14x measured); opencode → server-side
  `--chat-template-kwargs '{"enable_thinking":false}'`.

## Codex branch — state as of 2026-09-05 evening (Fable 5.1 session)

Fork source now lives at `~/Desktop/Github/codex-fork` (clone of
mjwsolo/codex, branch `localcode`, commit 76383e78 + follow-ups). Build:
`cd codex-rs && PATH=~/.cargo/bin:$PATH cargo build --release --bin codex`
(~5-6 min incremental). Install: `cp target/release/codex .run/codex-localcode.new
&& mv` into `localcode-codex/codex-agent/.run/` (never overwrite in place).

DONE and verified in tmux on real models (gemma 12B ⇄ Qwen 3.8 27B):
1. **/model inside the TUI is localcode's picker.** `codex-agent/localcode_supervisor.py`
   owns llama-server and serves `/catalog`, `/quants`, `POST /select`, `/status`
   on a control port (8323+); the launcher exports `LOCALCODE_CONTROL_URL`;
   `tui/src/chatwidget/localcode_picker.rs` renders model → quant (★ from
   recommend(), fit glyph, size, tok/s, downloaded) and polls the switch into
   the status line. A real 14.3 GB download + switch on the same port +
   "Model changed to …" + a 4-token "pong" turn (thinking off) all verified.
2. Debrand: 200 → 35 branded strings in the binary (rest are multi-line
   literals and skill-sample docs). Sweep script pattern is in memory
   `full-debrand-no-prompts`. Welcome line, tips, placeholder, prompts,
   models.json, login pages all read localcode.
3. Nothing phones home: announcement fetch, GitHub/Homebrew update check,
   analytics default, Statsig metrics exporter all off in the fork.
4. HTTP 500 from llama-server now shows the server's own message (was the
   fake "high demand" line — it appeared when gemma emitted malformed
   tool-call JSON).
5. Bugs fixed: supervisor used `signal.pause()` (exits on SIGCHLD from the
   server it just stopped); `config.toml` had approval/sandbox/thinking keys
   INSIDE `[model_providers.localcode]` (never applied) — provider table is
   now last; launcher tested `-z ""` so the pre-launch picker always showed.

STILL TODO — codex:
- Loop breaker: with the plan gate on, a gemma run spent 70 iterations on ONE
  TypeScript error (importExport.ts) until the 90-min cap. Port localcode's
  identical-failing-call hard-stop nudge / investigation-spin detector into
  the fork's turn loop next to the plan gate (core/src/session/turn.rs).
- 35 leftover "Codex" strings (multi-line literals, ext/ crates, skill samples).
- `/app` (Desktop app) command still exists; description now says unavailable.
- Upstream tracking loop (PINNED sha + weekly rebase) still missing.
- Enter under tmux: number keys select immediately in codex list popups; an
  Enter sent afterwards accepts the NEXT view's highlighted row.

DONE 2026-09-06 (benchmark-driven, see localcode/dev/anki-benchmark-2026-09-06.md):
- Per-machine server command in all three launchers (server_cmd.py →
  localcode's llama_server_command): 131k ctx + q8_0 KV on this Mac, was a
  hardcoded 32k with no KV compression.
- codex fork 567f1309: plan-then-execute + open-todo completion gate ported
  from localcode agent/loop.py; compaction keeps requirements + plan verbatim.
  Profile enables [tools.update_plan] (codex ships it OFF) and disables
  view_image (llama-server /v1/responses rejects image tool outputs, HTTP 400).
- Sandbox network on (npm install failed ENOTFOUND headless).
- pi e03967e: bash guard blocks foreground dev servers (19-min hang) and adds
  a 600 s default timeout; approval prompts removed (fc93b60).

## TODO — pi branch (`agent-ts/`)

0. Port localcode's todo_write + open-todo gate as a pi extension
   (`localcode-todo.ts`): register the tool, add the plan-first rule to the
   system prompt, inject open todos each turn, re-prompt on early stop. pi has
   no plan state at all today; on the Anki task it stopped with a broken build.

1. `pi-bump.yml` has never run in CI (workflow_dispatch needs default branch).
   First run happens at merge; steps pass locally via `npm test`.
2. Launcher/TUI polish not yet signed off: hand-test `/model`
   two-level picker + a real multi-GB download UX end to end.
3. Stall recovery (`auto_nudge`, 265 real fires in old localcode logs):
   measure whether pi's loop needs it before porting (pi went 4/4 on builds).
4. Compaction on long sessions never exercised; multi-hour session unknown.
5. Vision models: mmproj sidecar download implemented, never live-tested.
6. Version bump to `0.4.0a1` pattern before ANY publish (wheel currently
   inherits 0.3.71 which PyPI already has).
7. `try.sh` startup prints pi's "Model scope: <14 raw names>" dump — ugly,
   from `--models` flag; suppress or restyle.

## TODO — opencode branch

1. Launcher picker updated to the real localcode picker (same
   `model_picker_cli.py`); not yet hand-tested.
2. In-TUI model list uses opencode's own selector: a flat 2-entry list from
   `opencode.json` (confirmed 2026-09-05). Needs the same supervisor +
   model→quant picker treatment as codex, in the opencode fork's TUI.
3. No journeys rerun since thinking-off flag added to journeys server line.

## How to verify anything here (the protocol that caught 10+ shipped bugs)

- Drive TUIs with tmux, never expect(1): `tmux new-session -d -x 110 -y 32`,
  `send-keys`, `capture-pane -p`. expect's Enter never registers.
- PASS = an artifact the agent never saw: run pytest yourself, run the built
  CLI yourself, `ls` the gguf, `strings` the binary. Never trust self-reports
  (incl. your own claims — "verified" means you executed it).
- Any benchmark: same warm server, direct connection (router proxy costs
  9-21s/task), A/B interleaved, thinking PROVEN off on every wire first.
- llama.cpp router: serves only LOADED models; fresh files need
  `GET /models?reload=1`; POST /models "download" is a silent no-op — never
  use it (pi branch downloads straight from HF; do the same elsewhere).
- `.localcode/` in a project = classic localcode state; pi config dir is
  `.localcode-agent` for this reason. Don't regress it.
- Terminal.app windows close on exit → always append `; exec zsh` when
  opening demo windows, or crashes look like flashes.

## Definition of done

Every user journey driven end to end in a real terminal on real local models,
by the agent, with screenshots/artifacts — then handed over to be felt by a person. They
will find what you missed; when he says "u sure?", the answer is to go break
your own claim before he does.
