# LocalCode Error Codes

Every user-facing error in LocalCode has a stable `Eccc` code.
This file is **generated from `src/localcode/errors.py`**;
don't hand-edit it. To add or change a code, edit the registry
then run `python -m localcode.errors --emit-docs > docs/ERRORS.md`.

## E1xxx — Setup / startup

### `E1001` — Server failed to start
- **Cause:** `system`
- **Remediation:** Run `localcode setup` or check ~/.local/share/localcode/server.log

### `E1002` — Server didn't come up in time
- **Cause:** `system`
- **Remediation:** Try again. If it persists, re-run `localcode setup`.

### `E1003` — Model file not found
- **Cause:** `user`
- **Remediation:** Run `localcode setup` to download the model, or pick another via /model.

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

### `E3101` — Model returned an early end-of-text
- **Cause:** `model`
- **Remediation:** Quantization artifact — the model stopped before emitting useful output. Retry.

### `E3102` — HTTP stream disconnected mid-response
- **Cause:** `system`
- **Remediation:** Network or llama-server hiccup. Retry the turn.

### `E3103` — Context window overflow
- **Cause:** `user`
- **Remediation:** Conversation got too long. /clear to start fresh, or wait for compaction.

### `E3104` — Model server crashed
- **Cause:** `system`
- **Remediation:** Check ~/.local/share/localcode/server.log — usually a memory pressure kill.

### `E3105` — Model timed out
- **Cause:** `system`
- **Remediation:** First-token latency exceeded threshold. Server may be loading the model still.

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

