---
title: Architecture
description: The pieces localcode is made of, from the chat screen down to the model.
---

```text
  chat screen (TUI)  -->  agent loop  -->  tools (read / edit / bash / search / MCP)
                             |
                             v
                 model server on your Mac (http://localhost:8081)
                             |
                             v
                     GGUF weights on disk
```

- **Chat screen.** Setup, the model picker and chat are screens in one terminal app.
- **Agent loop.** The model asks for a tool, the tool runs, the result goes back to the model. The task, its to-do list and its goal carry over between your messages.
- **Tools.** Read and edit files, search with glob and grep, run shell commands, check syntax, navigate symbols, launch apps, fetch from the web, and anything an MCP server adds.
- **Model server.** localcode starts a `llama-server` that ships inside the package and talks to it at `localhost:8081`. You can point `runtime.base_url` at a server on another machine instead; see [Network Boundary](/localcode/concepts/network-boundary).

## Built for small models

Local models make predictable mistakes, and the loop corrects for them:

- Malformed tool calls are repaired instead of failing the turn.
- Cut-off tool calls and reasoning loops are detected and recovered from.
- Hidden reasoning is off by default. Turn it on per model with `/thinking`.
- Edits are syntax-checked before anything runs.

## Memory safety

- A memory-pressure monitor stops the model server before your Mac starts swapping. The event log records it.
- `localcode unstick` recovers a stuck server without a reboot.
