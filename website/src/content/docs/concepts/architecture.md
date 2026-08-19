---
title: Architecture
description: The pieces localcode is made of, from the TUI down to the inference server.
---

## The stack

The default configuration, which is what the rest of this page describes:

```text
  Textual TUI  ──►  agent loop  ──►  tools (read/edit/bash/search/MCP)
                        │
                        ▼
        runtime.base_url  (default http://localhost:8081)
                        │
                        ▼
              llama-server started by localcode
              (llama.cpp fork + TurboQuant KV compression)
                        │
                        ▼
                  GGUF weights on disk
```

The arrow marked `runtime.base_url` is a configuration value, not a fixed
edge. Point it elsewhere — `LOCALCODE_BASE_URL`, or the key in
`config.toml` — and the agent posts completions to that address instead, with
no validation and no change anywhere in the UI. See
[Network Boundary](/localcode/concepts/network-boundary#inference-endpoint-the-one-that-moves-the-boundary).

- **TUI** — the product surface. Setup, mode choice, model picker and chat are
  all screens in one Textual app.
- **Agent loop** — rounds of "model emits tool calls → tools run → results go
  back". Turn state, todos and goal context carry across user messages.
- **Tools** — file reads and edits, glob/grep, shell, project checks, syntax
  checks, code navigation and symbol inspection, notebook edits, app launch,
  plus the two network tools and any MCP tools you have configured.
- **Inference server** — by default a `llama-server` binary shipped inside the
  wheel, which localcode starts and addresses on `localhost:8081`. localcode
  binds *its own* server there; it does not constrain `base_url` to loopback,
  so the client half of this arrow goes wherever that setting points. See
  [Network Boundary](/localcode/concepts/network-boundary).

## Built for small models specifically

Quantised local models fail in characteristic ways, and the loop has explicit
handling for them rather than assuming a frontier model:

- **Tool-call repair** — malformed JSON arguments and stray whitespace in tool
  names are repaired at dispatch instead of failing the round.
- **Recovery modes** — dedicated paths for truncated tool calls and reasoning
  loops, with their own exit reasons in the event stream.
- **Hidden reasoning is off by default** — it is opt-in per model via
  `/thinking`, and models with no reasoning channel say so instead of silently
  ignoring the setting.
- **Syntax checks before shell runs** — tree-sitter catches broken edits
  in-process.

## Memory and process safety

- A memory-pressure monitor watches the server and can terminate it rather
  than let the machine swap to a halt; the kill is recorded in the project
  event log.
- A multi-region mmap patch in the fork fixes a Metal OOM where llama.cpp's
  loader mapped an entire GGUF into one Metal buffer.
- `localcode unstick` recovers a wedged server without a reboot.

:::note[Preview stub]
This page is a condensed overview for the docs preview. A fuller treatment —
module-by-module responsibilities, the round policy and state machine, and the
context/compaction system — still needs to be written.
:::
