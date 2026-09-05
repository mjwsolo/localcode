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

## TODO — codex branch (`codex-agent/` + mjwsolo/codex fork)

1. **/model inside the codex TUI shows codex's GPT catalog.** THE open item.
   Rust work in the fork: the TUI's model selector reads the bundled
   models-manager catalog (`codex-rs/models-manager/models.json`,
   `built_in_model_providers`). Make it localcode's picker: family → quant,
   sourced from the active provider's `/models` endpoint + localcode catalog
   (grep `codex-rs/tui/src` for the selector: "Select model (opens selector
   UI)"). Interim launcher picker (`codex-agent/model_picker_cli.py`) already
   does the flow pre-launch — the in-TUI one must match it.
2. Debrand sweep was running when this doc was written: mechanical
   `\bCodex\b` → `localcode` in `codex-rs/{tui,exec}/src` (commit pushed;
   background build `bxjno6d53` may need finishing). After building:
   `strings target/release/codex | grep -c "OpenAI Codex"` must be 0, then
   `cp` to `codex-agent/.run/codex-localcode`. Sweep the OTHER crates'
   user-visible strings the same way (onboarding/login/core prompts).
   Watch for: doc-tests/tests comparing literals (we don't run them).
3. Model download in-TUI: none. Launcher picker downloads pre-launch; in-TUI
   there's nothing. Depends on (1).
4. Codex update-check/login surfaces: with env-key auth we never saw login,
   but audit `strings` for chatgpt.com / update URLs like we did for pi.dev
   on the pi branch, and neutralize in the fork.
5. No upstream tracking: add PINNED upstream sha + weekly rebase loop like
   `upstream-bump.yml` / `pi-bump.yml`, or the fork rots (see the
   llama.cpp 4-month-drift memory).
6. Enter-submit is flaky under tmux (kitty keyboard protocol). Fine on real
   keyboards; scripts must send Enter variants with delays.

## TODO — pi branch (`agent-ts/`)

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
2. In-TUI model list uses opencode's own selector — same critique as codex
   likely applies; check `/models` in its TUI against the localcode rule.
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
