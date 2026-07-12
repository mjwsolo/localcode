# LocalCode Ruthless Technical and Product Review

Date: 2026-07-12

Scope: compare LocalCode with Pi (`badlogic/pi-mono`), OpenCode, Codex, and Claude Code, while optimizing for private local inference on Apple Silicon with 16–128 GB unified memory.

## Executive verdict

LocalCode should not try to beat Codex or Claude Code by copying their feature count. Its defensible advantage is vertical ownership of local inference: GGUF selection, Apple Silicon tuning, compressed KV cache, memory-pressure recovery, deterministic context reduction, and offline execution. Pi and OpenCode delegate most of that responsibility to providers; Codex and Claude Code use hosted inference.

That advantage is real, but the current release has four ship-blocking trust-boundary failures:

1. Default `auto_edit` can write outside the repository without approval.
2. Bash auto-approval is based on raw text prefixes and is not containment.
3. The inference-server executable download retries with TLS verification disabled and performs no artifact verification.
4. Server cleanup may kill unrelated processes based on a stale PID or occupied port.

Fix those before adding features. The next priority is proving the inference claims with reproducible benchmarks. Current policy is more sophisticated than current evidence.

## Competitive position

| Dimension | LocalCode | Pi | OpenCode | Codex | Claude Code |
|---|---|---|---|---|---|
| Local inference | Vertically integrated llama.cpp fork, model catalog, KV and memory policy | Connects to providers/local endpoints | Connects to providers/local endpoints | Hosted-model-first | Hosted-model-first |
| Core design | Large policy-rich Python loop | Minimal core, extension-heavy | Modular provider/client architecture | Rust CLI with OS sandbox | Mature hosted terminal agent |
| Context | Deterministic aging, spill, ledger, compaction | Strong structured compaction and session branching | Session compaction | Automatic management | Automatic compaction and memory |
| Safety | Application-level permissions; hard boundary is incomplete | Intentionally minimal; extensions/containers | Pattern permissions; limited hard containment | OS sandbox plus approvals | Permissions plus sandbox and hooks |
| UI strength | Hardware-aware model picker and runtime status | Fast, minimal, extensible TUI | Polished provider/model/session workflows | Clear progress and approval UX | Mature permissions, sessions, commands |
| Best lesson to borrow | N/A | Minimal core and structured session topology | Modular configuration and session UX | Separate consent from containment | Granular permissions and lifecycle polish |

Primary sources: [Pi coding agent](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md), [Pi compaction](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/compaction.md), [OpenCode agents](https://opencode.ai/docs/agents/), [Codex repository](https://github.com/openai/codex), [Claude Code overview](https://docs.anthropic.com/en/docs/claude-code/overview).

## P0: ship blockers

### 1. Repository containment is not real

`LocalCodeApp` creates `PermissionManager()` without a project root, while the default `auto_edit` mode session-approves writes and edits ([app.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/app.py:207), [autonomy.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/autonomy.py:49)). File tools accept absolute paths and traversal because joining an absolute path discards the repository root ([write_file.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tools/write_file.py:116)). Even when a root is supplied, string `startswith` incorrectly accepts sibling paths such as `/repo-evil` for root `/repo` ([permissions_v2.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/permissions_v2.py:96)).

Subagents compound this: they resolve and invoke raw executors directly, bypassing the parent approval path, and `general-purpose` agents receive write/edit/bash tools ([agent.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tools/agent.py:79), [agent.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tools/agent.py:143)).

Required fix:

- Pass the canonical repository root everywhere.
- Centralize path resolution using `Path.resolve()` and `is_relative_to()`/`os.path.commonpath`.
- Reject absolute paths, traversal, sibling-prefix paths, and symlink escapes.
- Recheck containment immediately before mutation.
- Route parent, subagent, MCP, skill, and headless tool calls through one authorization/execution path.
- Default subagents to read-only until this lands.
- Add an OS-backed macOS Seatbelt workspace sandbox. Approvals express consent; the sandbox provides containment.

### 2. Bash “safe prefixes” are composition-blind

Commands are auto-approved when their raw text starts with a known prefix ([permissions_v2.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/permissions_v2.py:238)). Shell operators, redirections, substitutions, and secondary commands make prefix classification unsound; a blacklist cannot model shell semantics.

Required fix:

- Remove raw-prefix auto-approval.
- Parse and classify every independent command segment and redirection, or ask for every shell command.
- Execute commands in the same workspace sandbox with network denied by default.
- Test separators, pipes, redirects, substitutions, interpreters, and reads of secret paths.

### 3. Executable download fails open

Bootstrap downloads a release binary, retries certificate failures using `ssl._create_unverified_context()`, and marks the result executable without a digest or signature check ([bootstrap.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/bootstrap.py:218), [bootstrap.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/bootstrap.py:252)).

Required fix: remove the TLS bypass; verify a signed, versioned SHA-256 manifest; download to a unique temporary path; verify before atomic install and `chmod`; publish provenance/attestation.

### 4. Process ownership is not verified

Shutdown trusts stale PID data and may kill a reused process group. It also discovers every PID on the preferred port and SIGKILLs it ([server_manager.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/server_manager.py:652), [server_manager.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/server_manager.py:720)).

Required fix: persist and verify PID, process start time, executable path/hash, command line, and an instance nonce. Never kill by port alone; choose another port when the occupant is not provably owned. Use SIGTERM before bounded SIGKILL.

## P1: inference correctness and proof

### Context overrides bypass the safety cap

`_target_num_ctx()` returns an explicit override before `_kv_aware_ctx_ceiling()` runs ([runtime.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/runtime.py:981)). A mistaken 128K/256K override can therefore bypass the claimed universal protection on 16 GB hardware.

Apply the ceiling to every request. Require an explicitly unsafe environment flag to bypass it.

### TurboQuant is missing from KV accounting

The production `turbo4` format is absent from `KV_DTYPE_BYTES`; unknown types disable model-specific clamping ([model_config.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/model_config.py:97), [runtime.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/runtime.py:820)). Encode measured architecture-specific bytes/token and recurrent/checkpoint overhead. Unknown combinations should fail conservatively unless a validated profile exists.

### Model “fit” is file-size fit, not working-set fit

Recommendation accepts weights up to 55% of physical RAM ([models_catalog.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/models_catalog.py:363)). It omits KV at the chosen context, compute buffers, checkpoints, vision projector, current pressure, and OS reserve.

Build one estimator shared by catalog, picker, and runtime. Recommend `{model, quant, context, vision, expected_peak, headroom}`, not a model alone.

### 256K is experimental, not a normal default

At 256K, checkpoints are disabled to avoid a known Metal wired-memory SIGKILL, and the comment labels the mitigation unverified ([runtime.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/runtime.py:569)). Default 96/128 GB systems to validated 128K; expose 256K as experimental until long-session soak testing passes.

### Thermal policy is advisory only

Thermal caps are computed but explicitly do not alter runtime behavior ([thermal.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/thermal.py:22), [thermal.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/thermal.py:98)). Apply caps before launch and between rounds, and show active throttling in the UI. Test MacBook Air separately from actively cooled Pro/Max machines.

### Add a real hardware benchmark matrix

Existing targeted policy tests passed, but they do not establish real-model performance. Record cold/warm load, TTFT, prompt-eval and decode throughput, peak resident/wired/compressed/GPU memory, checkpoint peaks, cache reuse, context shifting, 10/30/60-minute thermal curves, quality loss by weight/KV quant, and forced-pressure recovery. Record chip, RAM, OS, power mode, binary commit, GGUF hash, and ambient/battery state.

## P1: agent architecture

### Local subagents currently cost more than they add

They reuse the same runtime, run synchronously, carry no parent state, and may use twelve extra model rounds ([agent.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tools/agent.py:123)). On 16 GB this adds serial prefill/decode rather than independent intelligence.

Add a delegation admission gate: delegate only when expected parent-context savings exceed delegated prompt/output cost. Use one read-only explorer at a time on 16/32 GB and cap it at 4–6 rounds.

### The behavioral loop is too entangled

The main loop owns goals, reasoning, churn, reread recovery, completion gates, app launching, stages, telemetry, and dynamic schemas ([loop.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/agent/loop.py:41)). Keep deterministic hardware and safety controls, but split behavioral policies into observable components with explicit inputs/outputs and ablation tests. Borrow Pi’s minimal-core principle, not its lack of containment.

### Context strength needs session topology

Deterministic aging and the progress ledger are strong. Add Pi-like structured compacted state and session branches/forks: goal, constraints, decisions, read/modified files, test results, and unresolved risks. Do not solve this by increasing context windows.

## P1/P2: backend hardening

- Remove `trust_remote_code=True` for embeddings or pin and review an exact revision; prefer GGUF/ONNX and make embeddings lazy on 16 GB ([embeddings.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/embeddings.py:67)).
- Pin installed skills by digest/revision, validate size/frontmatter, preview diffs, record provenance, and separate installed from enabled ([skills.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/skills.py:130)).
- Make pre-tool hook timeout/error fail closed, avoid `shell=True`, require trust for project hooks, and sandbox hooks ([hooks.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/hooks.py:142)).
- Add explicit MCP shutdown/join and test repeated connect/disconnect RSS/process counts ([mcp/_bridge.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/mcp/_bridge.py:25)).
- Add history retention, redaction, restrictive permissions, delete/export, and ephemeral mode. SQLite WAL/locking itself is sound ([history.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/history.py:91)).
- Replace the fixed 11 GB model-switch threshold with incoming-model peak working set and current pressure ([server_manager.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/server_manager.py:240)).

## UI/UX review

### What is already good

- Setup is resumable and always offers a visible exit ([setup.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tui/screens/setup.py:19)).
- Errors have stable remediation and keep raw logs out of the primary UI ([errors.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/errors.py:68)).
- The model picker shows RAM fit, download state, approximate speed, recommendation, and current model ([model_picker.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tui/screens/model_picker.py:327)).
- The status bar exposes server, context headroom, thinking mode, task stage, and model ([chat.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tui/screens/chat.py:1707)).
- Slash commands are discoverable and unknown absolute paths are not misclassified as commands ([chat.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tui/screens/chat.py:693)).
- Targeted TUI/model-picker/input/output suite: 48 passed.

### Ruthless UX findings

1. **Every launch forces model selection and setup.** Returning users cannot reach the last healthy session/model directly ([tui/app.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tui/app.py:297)). Default to last-known-good, show a brief cancellable warm-up, and keep `/model` for switching. First launch should still require an explicit download choice.
2. **Permissions are presented as an unsafe binary toggle.** `/permissions` flips between “ask before commands” and “full auto, no questions asked” ([chat.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tui/screens/chat.py:2204)). Replace this with granular read/edit/command/network scopes, persistent project choices, and an always-visible mode indicator.
3. **The picker speaks quantizer jargon before user outcomes.** `UD-Q4_K_XL`, fit glyphs, and estimated tok/s serve experts, but new users need “recommended”, expected coding quality, usable context, peak memory/headroom, and download time first. Put raw quant details behind an advanced view.
4. **Estimated speed looks more authoritative than it is.** Keep `~tok/s` only after a one-minute local calibration or label it clearly as a rough estimate. Never rank primarily by it.
5. **Session continuity is hidden.** `--resume` exists, but no obvious in-product session picker/branch/fork workflow matches Pi, OpenCode, Codex, or Claude Code. Add `/resume`, `/sessions`, `/fork`, and a startup “continue last session” path.
6. **Approval summaries can hide load-bearing shell structure.** Multiline commands are collapsed and overflow is ellipsized ([chat_log.py](/Users/marcsolomon/Desktop/Github/localcode/src/localcode/tui/widgets/chat_log.py:1699)). Show parsed command segments, redirections, affected paths, and network intent before approval.

## Recommended defaults by RAM

| RAM | Model strategy | Context | Agent/runtime policy |
|---:|---|---:|---|
| 16 GB | Smallest validated coding quant; Gemma 4 12B Q4 baseline | 32K default; 64K opt-in validated profile | One inference server, vision/embeddings off unless needed, no subagents by default, early compaction, preserve 3.5–4 GB for macOS |
| 32 GB | Validated Gemma 26B-A4B IQ3 or Qwen 35B-A3B IQ2 | 64K | One bounded read-only specialist, optional lazy embeddings, measure Air vs Pro thermals |
| 64 GB | Quality quant such as Gemma 26B-A4B Q8 when peak fit is measured | 128K | Limited sequential delegated analysis; quality over context inflation |
| 128 GB | Best validated Qwen/Gemma quality quant | 128K default; 256K experimental | Concurrency must be memory-bandwidth/KV-budget aware; retain large OS/app reserve |

## Keep / change / remove

Keep: Apple Silicon specialization, RAM-aware catalog, asymmetric compressed KV, deterministic pre-compaction, pressure monitoring, single-server ownership concept, context/status visibility, resumable downloads, stable error remediation, SQLite WAL/locking.

Change: central authorization and sandboxing; peak-working-set recommendation; measured hardware profiles; bounded delegation; structured session forks; granular permissions; last-known-good startup; trusted skill/hook/MCP lifecycle; reproducible benchmarks.

Remove: TLS verification bypass; `trust_remote_code=True`; raw-prefix shell approval; kill-by-port; prefix-string path containment; writable subagents until centralized authorization; normal-default 256K claims; file-size-only “fits” claims; forced picker on every launch.

## Ordered implementation roadmap

1. P0: canonical path containment, central authorization, and read-only subagents.
2. P0: remove safe-prefix shell approval and add a macOS workspace/network sandbox.
3. P0: authenticated executable downloads.
4. P0: verified process ownership; never kill by port alone.
5. P0: adversarial regression suite for paths, shells, symlinks, delegation, PID reuse, and downloads.
6. P1: safe context overrides and measured TurboQuant accounting.
7. P1: canonical peak-working-set estimator shared by picker and runtime.
8. P1: real-model Mac benchmark harness; publish validated profiles.
9. P1: 128K default for high-RAM systems; 256K experimental.
10. P1: last-known-good startup, granular permissions, and session picker/forks.
11. P2: split behavioral policies and add ablation tests.
12. P2: harden skills, hooks, embeddings, MCP lifetime, and history privacy.

## Evidence status

- UI suite: 48 passed.
- Inference policy/model/thermal suites: passed with 10 environment-dependent skips.
- No code was modified by reviewers; only this report and the team roster were created.
- Competitor claims are limited to public first-party repositories and documentation; proprietary internals were not inferred as fact.
