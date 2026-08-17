# Changelog

All notable changes to LocalCode will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.3.51 — 2026-08-17

### Fixed

- **An empty or malformed stream chunk no longer crashes the turn with "Lost
  connection to the model server".** The SSE reader stripped the `data: ` prefix
  then called `json.loads(line)` directly — so an empty `data:` payload, a
  keepalive comment (`: ...`), or a whitespace-only line raised `Expecting value:
  line 1 column 1 (char 0)`, which aborted the whole stream and surfaced as a
  connection loss (seen live on a large-context Muse Glimmer Q8 run with ~19k-token
  prompts). The reader now skips empty/comment/non-JSON chunks and keeps reading;
  a single hiccup can't kill the turn.

## 0.3.50 — 2026-08-17

### Fixed

- **`read_file` on a directory returns the listing instead of erroring.** Live
  traces showed the model routinely calling `read_file` on a folder (confusing it
  with `list_files`), getting `IsADirectoryError`, and burning a whole round
  before retrying — painful on slow models (~40-90 s/round). It now hands back the
  directory contents plus a "use list_files for directories" nudge, so the round
  is productive.
- **System prompt discourages over-exploration.** Live logs showed the agent
  walking an entire existing codebase file-by-file (15+ reads/lists across 20+
  minutes) with zero writes. The prompt now says: explore only the files the task
  needs, and once you've seen enough (a handful of files) START WRITING — several
  reads/lists in a row without a write means you're over-exploring.

## 0.3.49 — 2026-08-17

### Fixed

- **The agent stops re-reading files it already has.** Real session traces showed
  ~68% of all `read_file` calls were re-reads of a handful of actively-used files
  (one `App.tsx` read 40× after being written), making long coding tasks feel
  read-heavy instead of write-heavy. Root cause: context aging treated `read_file`
  results as "replayable" and aged them out after the recent window (while write/
  edit diffs are never aged), so a file the model read fell out of context and it
  re-read it — and the system prompt told it to. Fix:
  - Context aging now **protects the latest read of the ~5 hottest (most-recently-
    read) files** from being aged out, mirroring how write/edit results are kept —
    bounded so prefill/TTFT stays capped.
  - System prompt: explicit "you already have a file's content after reading OR
    writing it — do NOT re-read to verify"; prefer a complete `write_file` /
    `multi_edit` over long read→edit→read loops; batch changes and verify ONCE
    instead of rebuilding after every edit.

## 0.3.48 — 2026-08-17

### Added

- **Approval prompt now has a `4` = "stop asking" option.** On any permission
  prompt, pressing `4` approves the current action AND turns off approval prompts
  for the rest of the session (flips autonomy to FULL_AUTO — everything
  auto-approved), the same state `/permissions` toggles. Re-enable prompts with
  `/permissions`. Prompt hint and the invalid-key reminder updated to show it.

## 0.3.47 — 2026-08-16

### Fixed

- **A reasoning cap trip no longer throws the whole turn away.** When the model
  reasoned past the per-round cap *without a detected loop* (e.g. a slow dense
  model doing a big planning pass), the turn hard-failed with "Stopped: model
  reasoning exceeded the per-round cap" and produced nothing. Now a cap trip is
  handled exactly like a detected loop: the aborted (empty) round is dropped and
  re-run with thinking OFF (up to the recovery budget), so the model **acts**
  instead of planning forever; only if it keeps over-reasoning does the turn end,
  with an honest message. Observed on Qwen 3.8 27B Q8: 10 min of legitimate
  planning, then total failure.
- **The reasoning time cap was tuned for fast models and aborted slow ones
  mid-thought.** It assumed ~50-75 tok/s; a dense model at ~17 tok/s hit the 600 s
  cap at only ~5.7k reasoning tokens — well under the 80k-char runaway limit, i.e.
  it aborted *normal* reasoning. Raised to 1800 s so the speed-independent char
  cap (~20k tokens) and the ~1 s structural loop detector remain the real guards.

### Notes

- **Hidden thinking is off by default** (`internal_thinking_mode = "off"`; a fresh
  config resolves to off). Turn it on per-session with `/thinking auto`. Reasoning
  models (Qwen 3.x, Muse Glimmer) are much faster to first action with thinking
  off, especially the slower dense quants.

## 0.3.46 — 2026-08-16

### Fixed

- **Live "thinking" now streams word-by-word instead of lurching a whole line at
  a time.** The reasoning stream (`stream_thinking`) only flushed on a newline,
  so the on-screen thinking froze until a line completed, then snapped the whole
  line in at once — very visible on slow/dense models (e.g. Qwen 3.8 27B at ~17
  tok/s, where a line takes ~2 s). The answer stream already flushed on word
  boundaries; thinking now does the same, rendering the trailing partial line in
  place (pop-and-rewrite, scoped to one logical line, guarded by an at-tail
  check) and committing to history on newline. Verified with the real-widget
  Textual pilot + unit tests. Note: this is a smoothness fix; raw generation
  speed is still governed by the model — dense models (Qwen 3.8, Muse Glimmer)
  decode slower than the ~3B-active MoEs (Qwen 3.6, Gemma MoE).

## 0.3.45 — 2026-08-16

### Added

- **Meta Muse Glimmer 30B support.** Meta's multimodal agentic model built for
  local deployment (Meta Superintelligence Lab, Aug 2026; Apache 2.0) — multi-step
  reasoning, tool use, failure recovery, and vision in one 30B model. Uses the
  unsloth GGUF (`unsloth/Muse-Glimmer-30B-GGUF`, UD-Q4_K_XL, 15.9 GB) with the
  BF16 perception encoder for image input. Its `muse_glimmer` architecture isn't
  in the TurboQuant server, so — like North-Mini-Code and DiffusionGemma —
  LocalCode builds a dedicated stock `llama-server` from current llama.cpp (PR
  #26841) on first use (one-time, ~5-12 min, cached) and serves it with stock
  flags + `--jinja` (required by its chat template). Pickable in the model
  picker; experimental (not auto-recommended — runs without TurboQuant KV
  compression). 32 GB+ unified memory recommended.

  Verified live on the rebuilt server: loads (multimodal), coherent text and
  code, correct arithmetic, native tool-calling via `--jinja`, vision (described
  a test image), and a full multi-round agentic task through LocalCode's runtime
  (write + read + todo across ~8 rounds, 73k-token accumulated context,
  `model_done`). New `ensure_muse_server` builder + `muse_server_binary` config +
  runtime routing; catalog/test guards updated.

## 0.3.44 — 2026-08-16

### Fixed

- **Qwen 3.x sessions no longer crash with "Lost connection / HTTP 500" after
  the first few turns.** Each round the agent appends an ephemeral context block
  (ledger + filesystem state + todo list) as a trailing `system` message. Qwen
  3.x's chat template raises `System message must be at the beginning` on any
  non-leading system message, which llama-server returns as HTTP 500 — so once
  that block became non-empty (a real multi-turn session with file changes), it
  fired **every round** and ended the turn with E3102. Qwen-family models now
  receive that block as a `user` turn instead (verified live: identical content
  returns 500 as `system`, 200 as `user`); Gemma keeps `system` (a user turn
  made it re-greet each round). Regression test in
  `tests/test_ephemeral_context_role.py`. This was latent in 0.3.42's Qwen 3.8
  support — single-tool smoke tests never populated the block, so it only
  surfaced under real multi-turn use.

## 0.3.43 — 2026-08-16

### Fixed

- **Setup screen no longer claims "one-time download" for a model that's
  already on disk.** The model step used a binary-GiB size threshold
  (`size_gb * 1024**3`) to decide "downloaded vs partial", but the catalog's
  `size_gb` is decimal GB — so a fully-present model read ~7% undersized and
  was labelled "Resume model (~38.5 GB)" with a misleading "one-time download ·
  cached for future launches" caption every launch. It now reuses the canonical
  `_is_complete_download` check (decimal GB, 3% tolerance): a present model
  shows "Model ready (~N GB)" and a "loading into memory" note; the
  download-framing captions only appear when a download is actually pending.

## 0.3.42 — 2026-08-16

### Added

- **Qwen 3.8 27B — now fully supported.** 0.3.41 removed it because the bundled
  server couldn't load the architecture; this release fixes the TurboQuant fork
  so it loads and runs. Qwen 3.8 is a dense 27B hybrid attention + Mamba-2 SSM
  model with a Multi-Token-Prediction (`nextn`) layer. The `qwen35` loader now:
  - reads `nextn.predict_layers` and treats only the real transformer blocks as
    attention/SSM layers (the trailing MTP block is no longer misread as an SSM
    layer — that was the `blk.64.ssm_conv1d.weight` crash);
  - registers and loads the four `nextn.*` tensors (classified as per-layer
    repeating tensors, so their `blk.N.nextn.*` names resolve);
  - runs the forward pass over the transformer stack only, skipping the MTP block.

  Verified with live runs on the rebuilt static server: loads (health 200,
  arch `qwen35`, 65 layers), coherent text and code generation, correct
  arithmetic, tool-calling (`list_files` emitted in Qwen/Hermes format), and
  vision (CLIP mmproj loads; correctly described a test image). A Gemma 4 12B
  regression run confirms existing models still load on the same binary.
  Catalog entry: `Qwen3.8-27B-UD-Q4_K_XL` (17.9 GB Q4, 32 GB+ Macs), with the
  F16 mmproj for image input.

## 0.3.41 — 2026-08-16

### Reverted

- **Qwen 3.8 27B removed.** 0.3.40 added it to the catalog, but a live load test
  showed the bundled TurboQuant server **cannot load it**: Qwen 3.8 is a hybrid
  attention+SSM model with a Multi-Token-Prediction (`nextn`) layer that the
  server's `qwen35` loader doesn't handle — it treats the MTP layer as an SSM
  layer and crashes with a missing-tensor error (`blk.64.ssm_conv1d.weight`).
  Rather than ship a model that fails to load, the entry is removed until the
  server supports the architecture.

### Added

- Catalog consistency test: every downloadable `ModelChoice` must map to a
  `MODEL_GROUPS` picker entry, so a future catalog addition can never again be
  invisible in the model picker.

## 0.3.40 — 2026-08-15

### Model catalog

- **Qwen 3.8 27B** — added Unsloth's new `Qwen3.8-27B-GGUF`: a **dense**
  vision-language model (image input via the F16 mmproj), **thinking on by
  default**, native **262K** context. Two curated quants — **UD-Q4_K_XL**
  (17.9 GB, ≥32 GB Macs) and near-lossless **UD-Q8_K_XL** (31.5 GB, ≥64 GB).
  It runs on the bundled TurboQuant server's native `qwen35` dense architecture
  (turbo4 KV + context checkpoints), and automatically gets Qwen's vendor-optimal
  sampler (temp 0.6 thinking / 0.7 instruct, top_k 20, min_p 0), the Qwen
  tool-calling adapter, and the chat-template thinking budget — full parity with
  the existing Qwen 3.6 entries.

## 0.3.39 — 2026-08-09

### Fixed

- **The reasoning ("thinking") indicator no longer lingers after you press Esc.**
  Esc-to-interrupt cancels the turn worker, which lands in `WorkerState.CANCELLED`
  — but the worker-state handler only cleaned up on `SUCCESS`/`ERROR`, so a cancel
  never finalized the turn: the animated `◆ thinking…` indicator kept spinning
  and input stayed wedged (`_agent_busy` stuck `True`). Cancelled agent turns now
  finalize like any other, so the indicator disappears and input is restored.

### Docs

- Dropped the redundant "Intel Mac — Not supported" row from the tested-hardware
  table; the Apple Silicon requirement is already stated under Requirements.

## 0.3.38 — 2026-08-09

### Fixed

- **Honest headless exit codes (for real this time).** `localcode run --json`
  decided its exit status from `emitter.last_turn_end`, but `turn_end` is
  emitted to the events file, not through the `OutputManager` callback the
  emitter listens on — so that dict was always empty and *every* headless run
  exited `0` / `status:"ok"` / `reason:"completed"`, even on `stall_exhausted`,
  `max_rounds`, `blocked_question`, or an errored loop. CI and eval consumers
  keying on the exit code treated every failure as a pass. The exit decision
  now reads the completion status the loop persists on the app each turn.
- **`glob` no longer hides real files behind noise directories.** The
  `.git` / `node_modules` / `.venv` / `__pycache__` exclusion ran *after* the
  100-item cap, so in a repo whose 100 most-recently-modified matches were
  dominated by `node_modules/` the tool returned truncated results — or "No
  matches" — while real source sat just past the cut. Exclude first, then cap.
- **Protocol outcome: a missing terminal field no longer becomes the string
  `"None"`.** `outcome_from_parse` used `str(x.get(k) if x else "") or ""`; a
  `result` event without an optional field (`final_text` is optional) produced
  the literal `"None"` (truthy, so the trailing `or ""` couldn't rescue it),
  polluting every normalized `RunOutcome`. Coalesce to `""` before `str()`.

## 0.3.37 — 2026-08-01

### Fixed
- **Calmer error UX.** A tool returning an error is normal, recoverable agentic
  work (the model reads it and adjusts) — but LocalCode painted every one in
  alarm-red, so routine successful runs *looked* like they were failing. Now:
  guardrail steers (read-before-edit / overwrite-staleness redirects) render as
  a quiet neutral note, other recovered tool errors render calm amber, and red
  is reserved exclusively for terminal, turn-ending failures.

### Docs
- README tested-hardware table adds the M5 (M5 Max, 128 GB) primary-dev row and
  notes that Linux is CI/dev-only while Apple Silicon (Metal) is the supported
  target.

## 0.3.36 — 2026-07-21

### Added
- **Capability-aware reasoning requests.** A typed model/provider registry
  (`reasoning_capabilities.py`) now drives request construction: llama.cpp gets
  chat-template thinking controls, OpenAI-style providers get `reasoning_effort`,
  the sampler thinking-budget is only sent for templates that can enforce it, and
  diffusion models can no longer accidentally enable a reasoning channel they
  don't have.
- **Compaction protocol validation.** Compaction now refuses to emit a transcript
  with a dangling tool call or an orphaned tool result — it returns the original
  messages instead, preventing a class of wedged-model states.
- **Configurable round cap + reasoning flags.** New `max_rounds` setting
  (0 = unlimited, as before; positive caps the loop for batch/eval), plus
  `run` CLI flags `--max-rounds`, `--thinking {off,auto,on}`, and
  `--thinking-budget` so a headless run can set the reasoning policy and budget
  without env vars. Explicit recovery-lifecycle events (`recovery_scheduled` /
  `recovery_completed` / `recovery_exhausted`, `generation_aborted`) for
  observability.
- **Honest headless exit codes.** `run --json` now returns a non-zero exit with
  status `incomplete` (and the loop's exit reason) when a turn ends without
  completing, instead of always reporting `ok`.
- **Bounded continuation** for output-limit truncation: valid visible text is
  preserved and the model is asked to resume from the exact cutoff (capped, no
  recap), while partial tool calls are still discarded and never executed.

### Changed
- Per-round recovery state is now a single immutable `NextRoundPolicy` snapshot
  instead of a parallel boolean, and `on` remains unconditionally on (the 0.3.35
  contract).

## 0.3.35 — 2026-07-21

### Changed
- **`/thinking on` is always-on again.** 0.3.34 made `on` skip the reasoning
  channel on "mechanical" stages. That was wrong on two counts: it silently
  broke the explicit opt-in (`on` should mean *on*), and the stage reflects the
  last completed action, not what the next round must decide — a round after a
  read/edit during `implement` may genuinely need reasoning, and gating it off
  killed that. Reverted: `on` forces thinking every round; `auto` remains the
  stage-aware policy. Reasoning-loop prevention stays where it belongs and does
  not lie about intent — the server-side thinking budget, the periodicity
  detector, and the verified no-thinking retry (all from 0.3.34).

## 0.3.34 — 2026-07-19

### Fixed
- **Cmd+C beep — the real cause.** Mouse capture was left ON in the primary
  `localcode` entry point (`app.run()` with Textual's `mouse=True` default),
  which disables the terminal's native selection, so Cmd+C on an empty native
  selection rang the terminal bell. The mouse-off default already existed in
  the `lc-tui` path but never reached `localcode`. Both entry points now launch
  with mouse capture off (opt back in with `LOCALCODE_MOUSE=1`).
- **Copy beep (OSC 52).** The clipboard fallback sequence was terminated with
  BEL (`\a`); a terminal that doesn't consume it rendered an audible beep. Now
  terminated with `ST` (`ESC \`).

- **No more "work at `$HOME`".** When launched outside any project (no
  `.git`/`package.json`/etc. marker), the target-location directive fell back
  to the user's home dir and told the model *"the project root is
  /Users/&lt;name&gt;, create all files under that root"* — so it tried to build in
  `$HOME` or fought the path the user named in the task. That directive is now
  suppressed in the unanchored case; the model follows the location from the
  request.
- **Scrollbars grey everywhere.** The model/version pickers (and any other
  scrollable widget) inherited Textual's blue default scroller. A global rule
  now makes every scrollbar the app's muted grey.

### Changed
- **Stronger tool-usage guidance in the system prompt.** When ONLINE, the model
  is now told to prefer its tools — `web_search`/`web_fetch` for current facts,
  versions, and library docs, plus its skills — over writing from memory. When
  OFFLINE, it's told to lean on local skills and installed tools rather than
  stalling on the missing network.

### Reasoning-loop recovery (small-model degeneration)
- **`on` no longer reasons on mechanical rounds.** With thinking set to `on`,
  the reasoning channel is now skipped on rounds explicitly marked mechanical
  (post-write implement / verify / running / complete) — there is nothing to
  reason about there, and forcing the degeneration-prone channel onto a
  mechanical round is what lets a small model spiral into a repetition loop.
  This bounds loop *incidence* at the policy layer (the `auto` policy already
  did this); reasoning stages and un-staged rounds still think, so `on` still
  means "reason deeply". Composes with the three layers below, which catch any
  residual loop.
- **Sampler-level thinking budget.** Every thinking request now sends llama.cpp
  a `thinking_budget_tokens` (8192) so the server forces the template's
  end-of-thinking transition and the *same* generation moves on to a tool call,
  instead of the reasoning channel running unbounded. This is the primary fix;
  the char/time caps are now compatibility backstops for templates whose think
  tags the server can't identify.
- **Degenerate-loop detector + auto-recovery.** A content-agnostic detector
  spots an exact repetition loop in the reasoning stream (a phrase cycling
  forever) within about a second, and the round retries once with thinking
  disabled — a decoding-mode switch that actually breaks the loop, instead of
  burning minutes to the length cap. Non-repeating runaways still hit the
  char/time cap with a clear message.

## 0.3.33 — 2026-07-19

### Sampling — vendor-optimal parameters per model (fixes the reasoning runaway)

The endless-reasoning hang was a **sampling bug**, not just a missing cap. localcode
was sending Qwen a too-wide `top_k` (40 — the server default, never overridden),
too-hot temperature, and `repeat_penalty = 1.05` — which Qwen explicitly warns
against because a repeat penalty **starves the EOS token**, so the model never
stops. `top_k`, `top_p`, `presence_penalty`, and `min_p = 0` were never forwarded
to the server at all.

Each model now gets its **official vendor-recommended sampler**, forwarded verbatim:

| Model | temp | top_p | top_k | min_p | presence_pen | repeat_pen |
|---|---|---|---|---|---|---|
| **Qwen 3.6** (thinking, coding) | 0.6 | 0.95 | 20 | 0 | 0 | 1.0 |
| **Qwen 3.6** (instruct) | 0.7 | 0.80 | 20 | 0 | 1.5 | 1.0 |
| **Gemma 4** | 1.0 | 0.95 | 64 | 0 | 0 | 1.0 |
| **North-Mini-Code** (Cohere) | 1.0 | 0.95 | 0 (off) | 0 | 0 | 1.0 |

Sources: the Qwen3.6-35B-A3B model card + Unsloth guide, the Gemma team's inference
config, and Cohere's North-Mini-Code card. The sampler is now the single source of
truth (the old "tuned but never sent" `_options` knobs and the `_coding_temperature`
0.25 cap that fought the vendor values are gone). A user-set temperature can still
only lower the vendor value.

**Both Google models are handled — deliberately differently.** Gemma 4 (above) runs
autoregressively through llama-server, so it needed explicit params (the server
default `top_k=40` was wrong). **DiffusionGemma** runs through `llama-diffusion-cli`
with the Entropy-Bound decoder, whose `--diffusion-eb-t-min/t-max/max-steps` all
default to **"from model metadata"** — i.e. the model ships its own recommended
values. So DiffusionGemma correctly inherits them and localcode does NOT override
them; hardcoding constants there would replace the model's own tuning. Same lesson,
opposite action: "trust the default" was the bug for llama-server, but is correct for
the diffusion CLI.

## 0.3.32 — 2026-07-19

### UI

- **Reasoning streams live.** The model's thinking now renders in the chat log
  as it arrives (dimmed, line by line, like Claude Code / Codex) instead of
  hiding behind a spinner and appearing only at the end. You can see it working
  every step — no more "9k tokens, no output, is it stuck?". The streamed lines
  are recorded in history so a resize or toggle doesn't wipe them.
- **Input scrollbar matches the app.** The chat input's scrollbar was a brand
  blue that clashed with the grey scrollbars everywhere else; it's now grey.

### Reliability

- **Runaway-reasoning guard re-enabled.** A model can loop in its thinking
  phase forever (observed: 29 minutes / 94.8k tokens on Qwen Q8, no output).
  The per-round thinking cap — disabled in April because its old bounds
  (90 s / 4000 chars) cut *legitimate* reasoning — is back with generous
  bounds (10 min / ~20k tokens) that only catch a true runaway, never normal
  thinking. On abort the user now gets a clear message telling them to try
  `/thinking off` or a faster model.
- **Turn-ending notices now show in the TUI.** `print_info` wrote only to
  stdout, so messages like the reasoning-cap abort were invisible in the TUI —
  a turn would just stop with no explanation. A new `notice` path emits an
  event the TUI renders.

### Model catalog

- **Removed the confusing QAT duplicates.** The Gemma QAT entries added earlier
  were single-quant repos that showed up as a second "Gemma 4 12B / 26B" next
  to the originals and browsed to just one row. Reverted to the clean Gemma
  lineup; the July-16 refreshed-weights advisory stays.
- **Quant browser hides sidecar files.** Vision projectors (`mmproj-*`),
  speculative-decoding draft heads (`mtp-*`), and files in subfolders no longer
  appear as tiny "0.5 GB" rows in the quant picker — only real, selectable
  model weights are listed.

## 0.3.31 — 2026-07-19

### Security

Hardening pass from a full security audit of the tool trust boundary. The
threat model: the local model's tool output is steerable by prompt injection
(a hostile README or source comment read into context), so the question is
what can execute without the user's consent.

- **Autonomy-independent safety hard-block.** A new pre-dispatch layer runs
  before every tool in every mode (including `full_auto` and headless) and
  cannot be overridden. It blocks only catastrophic shell with no legitimate
  use — `rm -rf /` (or `~`/`$HOME`), `mkfs`, `dd of=/dev/…`, `> /dev/sd*`,
  `chmod -R 777 /`, fork bombs, `> /etc/`, `wipefs` — from **both** `bash` and
  `background_process`, and refuses writes to credential/key files (`~/.ssh/*`,
  `~/.aws/*`, `id_rsa`, `authorized_keys`, `.netrc`, `credentials`,
  `/etc/shadow`). Matching is anchored and precise: it never fires on a command
  that merely *mentions* the text (`grep "DROP TABLE"`, `cat notes_about_mkfs.md`)
  and never on the project's own `tokenizer.py` / `api_keys.py`.
- **High-risk-but-legit commands now confirm instead of running silently.**
  `curl … | sh`, `git push --force`, and `sudo rm` are not hard-blocked (they
  have real uses) — they now prompt for approval, where before they ran with no
  prompt.
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
