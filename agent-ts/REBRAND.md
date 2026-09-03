# localcode on pi: everything left to change

Goal: `pip install localcode` → it just works. The user picks a model, downloads
it, and codes. They never see the upstream harness's name anywhere.

Status key: **done** (verified) · **written** (code exists, not verified) ·
**todo** · **needs patch** (no native hook; would be our first carried patch)

---

## A. Branding surfaces

| # | Surface | What the user sees now | Fix | Status |
|---|---|---|---|---|
| A1 | App name / title | terminal tab said `π - agent-ts` | `dist/package.json` → `piConfig.name` drives `APP_NAME`, `APP_TITLE` | **done** — tab now reads `localcode` |
| A2 | Config directory | `~/.pi/agent/` | `piConfig.configDir: ".localcode"` | **done** |
| A3 | Debug log, agent dir env | `pi-debug.log`, `PI_CODING_AGENT_DIR` | derived from `APP_NAME`, follows A1 automatically | **done** |
| A4 | Startup banner | `pi v0.84.4` + hint lines | `ctx.ui.setHeader()` in `localcode-brand.ts` | **written** — did not render from `session_start`; needs an earlier hook |
| A5 | Footer | built-in `0.0%/0 (auto) … unknown` | `ctx.ui.setFooter()` — brand pinned bottom-left, model/branch/status right-aligned | **written** — same hook problem as A4 |
| A6 | Startup hint text | *"Pi can explain its own features and look up its docs. Ask it how to use or extend Pi."* | hardcoded at `modes/interactive/interactive-mode.ts:1003`. Try `quietStartup: true` first; else replace via `setHeader` | **todo** — **needs patch** if the setting doesn't suppress it |
| A7 | System prompt | told the model it runs "inside pi" and pointed at pi's docs | `before_agent_start` replaces `customPrompt` | **written** |
| A8 | `/llama` command | exposes llama.cpp + upstream naming | hide it; fold load/unload/download into `/models` | **todo** |
| A9 | "No models available. Use /login…" warning | upstream copy, wrong advice for us | our first-run flow should fire before it; otherwise override | **todo** |
| A10 | Shipped `dist/docs/*.md` | error messages link to pi's docs files | stop copying them in `build.sh`; ship localcode docs or none | **todo** |
| A11 | `PI_*` env vars | `PI_OFFLINE`, `PI_BASE`, `PI_KEY`, ~10 more | hardcoded upstream; invisible to normal users | **todo (low)** — **needs patch** for a full scrub |
| A12 | `/help`, command descriptions | may name pi | audit the command list, override descriptions where ours | **todo** |

## B. First run — "pip install and it works"

**Verified**: the harness *does* start its TUI with zero models (it shows a
warning banner, it does not exit). My earlier claim that first-run had to live
in Python was wrong. So the whole flow can be native.

The flow to build:

1. `localcode` (Python) starts, finds no GGUF on disk, starts `llama-server` in
   router mode, execs the agent binary. **todo**
2. Extension `session_start`: no models on disk → open the `/models` picker
   automatically, with the ★ recommendation preselected. **todo**
3. User picks → download with visible progress. **written** (currently shells
   out to the `hf` CLI — see C1)
4. On completion: load into the router, refresh the model list, `setModel`,
   drop the user into the prompt. **todo** — see C2
5. Every later run: model already there, straight to the prompt. **todo**

## C. Functional gaps that block shipping

| # | Gap | Why it matters | Fix |
|---|---|---|---|
| C1 | Download shells out to the `hf` CLI | not a dependency we ship; breaks on a clean machine | call localcode's own downloader over a tiny local control socket, or implement the HF fetch in TS |
| C2 | Provider registers models **once** at startup | a model downloaded mid-session doesn't appear until restart | register a full `Provider` with `refreshModels` (upstream's own llama extension is the 1,453-line template) |
| C3 | No permission system | it will run `bash` without asking — unacceptable for a shipped product | `tool_call` hook can block; port `permissions_v2`, `execution_policy`, `injection_defense` |
| C4 | Router vs single-model | router gives switching + download; measured slower per turn | default single-model, switch to router only while browsing models |
| C5 | Model switching needs the server to already serve it | picker says "load it with /llama" — a leak of A8 | `/models` should drive the router load itself |

## D. Must not regress (localcode behaviour worth keeping)

- **thinking off by default** — done in the provider; this was worth ~2x runtime
- reasoning-cap trips must no-think-recover, never hard-fail
- 1800s time cap for slow models
- `thinking_budget_tokens` vs `reasoning_budget_tokens` endpoint split
- memory guard + thermal throttle (`before_provider_request` hooks)
- redaction, injection defense
- the curated catalog, RAM-based recommendation, HumanEval numbers, licenses

## E. Suggested order

1. **C3 permissions** — nothing ships without it
2. **B1-B5 first run** + **C1, C2, C5** — this is the "just works" promise
3. **A4, A5, A6, A9** — the branding the user actually sees on screen
4. **A8, A10, A12** — the leaks they'd find by poking around
5. **A11** — cosmetic scrub, likely a carried patch, do last

Items marked **needs patch** are the only places where a fork of upstream is
implied. Everything else is native: `registerCommand`, `ctx.ui.*`,
`setHeader`/`setFooter`, `before_agent_start`, `before_provider_request`,
`tool_call`, and `piConfig`.
