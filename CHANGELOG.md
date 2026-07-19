# Changelog

All notable changes to LocalCode will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.3.31 — 2026-07-19

### Security

Hardening pass from a full security audit of the tool trust boundary. The
threat model: the local model's tool output is steerable by prompt injection
(a hostile README or source comment read into context), so the question is
what can execute without the user's consent.

- **Autonomy-independent safety hard-block.** A new pre-dispatch layer runs
  before every tool in every mode (including `full_auto` and headless) and
  cannot be overridden. It blocks dangerous shell (`curl … | sh`, `dd`,
  `mkfs`, `rm -rf /`, fork bombs, force-push to main, `DROP DATABASE`) from
  **both** `bash` and `background_process`, and refuses writes to credential/
  key files (`~/.ssh/*`, `~/.aws/*`, `id_rsa`, `authorized_keys`, `.netrc`,
  `credentials`, `/etc/shadow`). Precise basename/segment matching, so editing
  the project's own `tokenizer.py` or `api_keys.py` is never affected.
- **`background_process` now goes through the approval gate.** Previously it
  ran `shell=True` on the raw model command with no prompt — the same command
  that `bash` would ask about. It is now confirmed like `bash`.
- **`suggest` (read-only) mode actually confirms writes.** File-write tools
  now prompt in suggest mode instead of executing silently.
- **Project hooks require explicit trust.** A repo's `.localcode/hooks.toml`
  runs shell at session start / on every prompt / before every tool. It is no
  longer loaded just because you opened the folder — that was remote code
  execution on `git clone && localcode`. Untrusted project hooks are disabled;
  review them with `/hooks` and enable with `/hooks trust` (re-prompts if the
  file changes). Global `~/.localcode/hooks.toml` stays trusted.
- **No more arbitrary `npm run` at the verify step.** The auto-verification
  that runs after a build no longer executes `npm run <script>` (an arbitrary
  shell string from the repo's `package.json`); it runs the `tsc --noEmit`
  binary directly, which executes no project code.
- **Download integrity pinning.** Catalog entries gained optional `revision`
  (pin to a commit SHA/tag instead of the mutable `main` tip) and `sha256`
  (verified after download; the file is deleted and rejected on mismatch).
- **`list_files` no longer un-hides `.env`.** It was the one dotfile the
  listing surfaced; now hidden like the rest so it isn't advertised to the
  model.

## 0.3.30 — 2026-07-19

### New features

- **`/delete` slash command** — remove downloaded models to free disk space.
  Bare `/delete` lists what's on disk with human-readable sizes; `/delete
  <number or name>` previews exactly which files would be removed and how much
  space that frees; only `/delete <target> confirm` actually deletes. The
  currently loaded model and in-flight downloads are refused with a clear
  message, partial downloads (`.part` files, undersized GGUFs, and hub resume
  state) are cleaned up, and a vision sidecar shared between quants of the
  same family is only removed with its last remaining model.

### Model catalog

- **Gemma 4 QAT entries** — Unsloth's quantization-aware-trained repos join
  the catalog: 12B QAT (6.7 GB, lightest full-quality pick for 16 GB Macs),
  26B-A4B QAT (14.3 GB, for 32 GB), and the catalog's first dense 31B entry
  (17.3 GB QAT, best on 48-64 GB). QAT weights are trained quantized, so Q4
  keeps near-BF16 quality.
- **Re-download advisory** — Google silently refreshed all Gemma 4 GGUF
  weights on 2026-07-16 (tool-calling and vision fixes, same filenames).
  Gemma catalog notes now flag that copies downloaded before then are stale;
  use `/delete` and re-download to pick up the fixes.
- **Delisted the non-QAT Gemma 4 12B Q4 entry** — superseded outright by the
  12B QAT entry (smaller file, near-BF16 quality). Already-downloaded copies
  keep working and the repo remains browsable; the curated picker just no
  longer offers the weaker duplicate. The 26B IQ3_S (only 26B that fits
  16 GB, measured 95.1% HumanEval), 12B BF16 (reference baseline), and Q8
  entries remain — their repos carry the refreshed 2026-07-16 weights.
- Fixed the DiffusionGemma browse note recommending a quant that does not
  exist in its repo (it ships plain Q4_K_M/Q5_K_M/Q6_K/Q8_0/BF16 only).

## 0.3.23 — 2026-07-12

### Agent harness

- Added a deterministic discover, plan, implement, verify, repair, and complete
  task state machine with conservative goal classification.
- Added hash-bound verification evidence that becomes stale when code, commands,
  or relevant environment values change. Unverified edits can no longer be
  reported as completed after the retry gate expires.
- Added adaptive reasoning and TTFT-driven hot-context replay limits for fast
  long-running work on local hybrid-memory models.
- Added failure-aware shell assessment that distinguishes exit status from task
  success, including masked failures, fallbacks, and unsafe pipelines.

### Tools and extensibility

- Added deterministic symbol, definition, and reference navigation.
- Added supervised background processes with stable IDs, ownership, durable
  logs, incremental polling, status, and explicit stopping.
- Added pre/post-tool, post-edit, pre-completion, and post-compaction hooks.
- Added progressive Agent Skills discovery across LocalCode, `.agents`, Claude,
  and OpenCode layouts while bounding catalog prompt cost.

## 0.3.0 — 2026-04-28

First open-source release.

### Licensing

- Switched from proprietary to **Apache 2.0**.
- Added Apache 2.0 license file, SPDX identifier in `pyproject.toml`, and OSI
  classifier.

### Agent loop

- **Loop split**: the 964-line monolith in `agent/loop.py` is now spread across
  focused modules: `goal.py` (intent inference), `streaming.py` (token/event
  collection), `tool_execution.py` (dedup + oversize guards), `tool_orchestration.py`
  (parallel dispatch), `turn_finalization.py` (telemetry + persistence),
  `hooks.py` (lifecycle hooks), `prompt_context.py` (system prompt assembly),
  `app_tasks.py` (build-app stage inference).
- **CLI → entrypoint**: `cli.py` renamed to `entrypoint.py` and rewritten to own
  argv parsing, subcommands (`config-init`, `setup`, `benchmark`, `status`,
  `models`, `unstick`), and GPU-unlock signaling (exit 42 → `sysctl iogpu.wired_limit`).

### New features

- **Deterministic app launcher** (`launcher.py` + `tools/launch_app.py`) detects
  `package.json` / `pyproject.toml` / static sites, picks a free port, starts
  the process, verifies localhost reachable, and records PID/port/URL in
  `process_registry.py`.
- **Task tracking**: `SessionState` carries `current_task` and `recent_tasks`
  with goal type, stage, port, completion status, and blocked reason. History
  DB gained 9 new columns (additive migration).
- **Adaptive thinking**: `should_use_thinking()` decouples runtime mode from
  internal thinking policy. Configurable via `LOCALCODE_INTERNAL_THINKING_MODE`
  or the `/thinking off|auto` slash command.
- **Dynamic skills**: `select_dynamic_skills()` injects targeted skill cards
  based on the last failed tool, then current-turn intent, with a soft prompt
  budget cap.
- **Destructive-write guards**: `write_file` rejects calls that would collapse
  a 60+ line source file to fewer than 20 lines. `multi_edit` simulates its
  edits before applying and rejects the same pattern.
- **Recovery modes**: when `tool_args_limit` fires repeatedly during a build,
  the loop escalates to `large_write` (encourage append/edit) and
  `large_write_final` (remove `write_file` from the schema entirely),
  preventing the model from looping on monolithic writes. Same `repeat_failed:<tool>`
  recovery applies generically when any tool fails the same call twice.

### Hardening

- **`clean_env()`** strips `MallocStackLogging*` and `MallocNanoZone` from
  subprocess environments. Threaded through `health.py`, `recovery.py`,
  `hooks.py`, `verification.py`, `launcher.py`.
- **Bash tool**: tree-output compression for results >20 KB, smarter detached
  server detection, port-listening check, startup-command hinting.
- **Read file**: default limit dropped from 2000 lines to 240 / 12 KB cap with
  explicit offset/limit hint when truncated, preventing accidental prompt bloat.

### Prompts

- `SYSTEM_PROMPT` split into `MINIMAL_CORE` / `BASELINE` / `TIGHT` variants.
- `sections.py` refactored to render 7 cacheable sections via a render cache —
  preparation for prompt caching.

### TUI

- Removed `/mode` and `/plan show|off` slash commands (planning is now an
  artifact, not a tool gate).
- Added `/thinking off|auto` to control the internal reasoning policy.
- Cost display replaced with in/out/total token counts.
- Task stage now shown in footer status bar.

### Tooling & ops

- New `scripts/analyze_events.py` — telemetry analyzer over `.localcode/events.jsonl`.
- New `scripts/audit_release.py` — flags generated artifacts (`sample_learning_app*/`,
  `logo/`, `*.log`) accidentally mixed with core changes before release.
- 13 new regression tests in `tests/test_agent_event_regressions.py` covering
  write-file refusals, completion gates, launcher heuristics, process registry
  round-trip, dynamic skill injection, read-file caps, bash backgrounding, and
  tool-fact extraction.

### Branding

- Added theme-aware logos (`docs/assets/logo/dark.png`, `docs/assets/logo/light.png`)
  rendered via `<picture>` + `prefers-color-scheme` in the README. GitHub, PyPI,
  and MkDocs honor the dynamic switch.

## 0.2.x and earlier

See `git log` for history. Key milestones:

- TurboQuant KV cache integration (asymmetric q8\_0-K + turbo4-V).
- Multi-region mmap patch (commit `3d66675b8`) — fixed Metal OOM by emitting
  one buffer per contiguous tensor run instead of a single span.
- `iogpu.wired_limit_mb` auto-unlock + LaunchDaemon.
- Anti-loop sampler forwarding, stall-aware history pruning, and prompt overhaul.
