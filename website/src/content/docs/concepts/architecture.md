---
title: Architecture
description: The pieces localcode is made of, from the TUI down to the inference server.
---

## The stack

This is the default setup. The rest of this page describes it:

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

- **TUI** - the main product interface. Setup, mode choice, the model picker, and chat are all screens in one Textual app.
- **Agent loop** - the model creates tool calls, the tools run, and the results go back to the model. Turn state, todos, and goal context continue across user messages.
- **Tools** - file reading and editing, glob/grep, shell commands, project checks, syntax checks, code navigation and symbol inspection, notebook editing, app launching, the two network tools, and any MCP tools you have configured.
- **Inference server** - by default, localcode starts its own `llama-server` (the binary included in the wheel) at `localhost:8081`.

## Built specifically for small models

Quantised local models often fail in predictable ways. The loop handles these problems directly instead of assuming it is using a frontier model:

- **Tool-call repair** - the dispatcher fixes malformed JSON arguments and extra spaces in tool names instead of failing the round.
- **Recovery modes** - separate recovery paths handle cut-off tool calls and reasoning loops. Each path has its own exit reason in the event stream.
- **Hidden reasoning is off by default** - you can turn it on for each model with `/thinking`. Models without a reasoning channel say so instead of silently ignoring the setting.
- **Syntax checks before shell runs** - tree-sitter finds broken edits in-process.

## Memory and process safety

- A memory-pressure monitor watches the server. It can stop the server instead of letting the machine become unusable from swapping. The project event log records the stop.
- A multi-region mmap patch in the fork fixes a Metal OOM caused when llama.cpp's loader mapped a whole GGUF into one Metal buffer.
