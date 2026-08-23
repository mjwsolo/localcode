# LocalCode Error Codes

Every user-facing error in LocalCode has a stable `Eccc` code.
This file is **generated from `src/localcode/errors.py`**;
don't hand-edit it. To add or change a code, edit the registry
then run `python -m localcode.errors --emit-docs > docs/ERRORS.md`.

## E1xxx — Setup / startup

### `E1001` — Server failed to start
- **Cause:** `system`
- **Remediation:** Restart LocalCode (setup re-runs automatically) or check ~/.local/share/localcode/server.log

### `E1002` — Server didn't come up in time
- **Cause:** `system`
- **Remediation:** Try again. If it persists, restart LocalCode.

### `E1003` — Model file not found
- **Cause:** `user`
- **Remediation:** Pick a model via /model — it downloads automatically.

### `E1004` — Backend not initialized
- **Cause:** `user`
- **Remediation:** Type a message to trigger backend startup.

### `E1010` — Insufficient memory to launch
- **Cause:** `user`
- **Remediation:** Quit other heavy apps (browsers, IDEs, Docker) and retry.

### `E1011` — Stuck llama-server from prior session
- **Cause:** `system`
- **Remediation:** Restart your Mac to clear the GPU-wait — auto-recovery couldn't unstick it.

## E2xxx — Tool dispatch

### `E2101` — Unknown tool
- **Cause:** `model`
- **Remediation:** The model emitted a tool name that isn't registered. Likely a quantization artifact.

### `E2102` — Malformed tool arguments
- **Cause:** `model`
- **Remediation:** The model emitted invalid JSON for the tool's arguments.

### `E2103` — Tool name had stray whitespace
- **Cause:** `model`
- **Remediation:** Auto-stripped at dispatch — the call still ran. Logged for telemetry.

### `E2104` — Missing required argument
- **Cause:** `model`
- **Remediation:** The model omitted a required parameter for the tool. Often retried in the next round.

### `E2110` — Tool denied by permission policy
- **Cause:** `user`
- **Remediation:** The user (or session policy) declined this tool call.

### `E2111` — Tool blocked by hook
- **Cause:** `user`
- **Remediation:** A configured hook in ~/.localcode/hooks.toml refused the tool call.

## E3xxx — Runtime / model

### `E3101` — The model stopped responding too early
- **Cause:** `model`
- **Remediation:** Try again. If it keeps happening, switch model with /model.

### `E3102` — Lost connection to the model server
- **Cause:** `system`
- **Remediation:** Send another message — the server auto-restarts. If it recurs on big builds, `/model` to a smaller quant.

### `E3103` — Conversation is too long for this model
- **Cause:** `user`
- **Remediation:** Type /clear to start a fresh conversation.

### `E3104` — The GPU couldn't run the model
- **Cause:** `system`
- **Remediation:** Quit big apps to free memory and try again. If it keeps failing, restart your Mac.

### `E3105` — The model is still loading
- **Cause:** `system`
- **Remediation:** Wait a few seconds and try again — the model takes a moment to warm up.

### `E3106` — macOS paused your model server to protect the system from running out of memory
- **Cause:** `system`
- **Remediation:** We auto-restart on your next message — usually no action needed. To prevent it: close memory-heavy apps (browsers with many tabs, IDEs with large projects, video editors), or `/model` to a smaller quant. Your Mac was at critical memory pressure during your last request, so the safety monitor freed memory by killing the server before macOS itself would have force-killed it.

### `E3107` — Diffusion model returned no usable output
- **Cause:** `model`
- **Remediation:** Auto-retried. For heavier agentic work, `/model` to a Gemma 26B-A4B or Qwen quant.

### `E3108` — Model collapsed into repeated junk tokens
- **Cause:** `model`
- **Remediation:** Known Gemma-4 llama.cpp bug. `/model` to Gemma 4 12B, or rebuild the server binary.

## E4xxx — Filesystem / git

### `E4101` — Path is a directory, not a file
- **Cause:** `model`
- **Remediation:** Use `bash mkdir -p <path>` to create directories; write_file is for files.

### `E4102` — Path not found
- **Cause:** `model`
- **Remediation:** Check the path. The model may have hallucinated it.

### `E4103` — Permission denied (filesystem)
- **Cause:** `system`
- **Remediation:** macOS may be sandboxing this directory. Try a path under your home dir.

### `E4110` — Git command failed
- **Cause:** `user`
- **Remediation:** Check `git status` for repo state; the failed command's output is logged above.

## E5xxx — User cancellation / loop-breakers

### `E5101` — Cancelled by user
- **Cause:** `user`
- **Remediation:** You typed 'stop' / 'cancel' / 'abort' during a running turn.

### `E5102` — Loop-breaker: same call 3× in a row
- **Cause:** `model`
- **Remediation:** Model was stuck. Tell it what you actually want, more concretely.

### `E5103` — Loop-breaker: too many tool calls in one turn
- **Cause:** `model`
- **Remediation:** Model was thrashing. Break the task into smaller steps.

### `E5104` — Loop-breaker: file edited too many times
- **Cause:** `model`
- **Remediation:** Model was oscillating. Read the file and tell it what's actually wrong.

## E9xxx — Wrapped unknown

### `E9001` — Unhandled exception in agent loop
- **Cause:** `internal`
- **Remediation:** Internal bug. Paste the exception type + message into a github issue.

### `E9002` — Unhandled exception in TUI
- **Cause:** `internal`
- **Remediation:** Internal bug. Restart localcode; if it recurs, file a github issue.

