---
title: Network Boundary
description: The network paths localcode can use, what triggers them, and what stays on your Mac.
---

**Inference runs on your Mac. It needs no API key, model provider, or account.**
A few features use the network, and each one starts with something you asked
for. This page lists every path.

## What stays on your Mac

- **Inference.** The interface sends chat completions to `http://127.0.0.1:PORT/v1`,
  a `llama-server` process started by localcode from the binary included in
  the wheel. The provider is fixed to that server; there is no setting that
  points inference elsewhere.
- **Your files, prompts and edits.** The agent reads and writes your working
  tree directly.
- **Sessions and logs.** Sessions live in `<project>/.localcode-agent/`. Server
  and supervisor logs live in the run directory. Nothing is uploaded.

There is no analytics endpoint, usage reporting, update check, or account.
Every localcode process binds to `127.0.0.1`.

## Where localcode uses the network

| # | What | Where it goes | When |
| --- | --- | --- | --- |
| 1 | **Model download** | `huggingface.co` | Only when you press Enter on a `Download` row in the picker. The size is shown first |
| 2 | **Quant listing** | Hugging Face repo API | When you open a model in the picker, to list the quants its repo ships |
| 3 | **Image projector** | `huggingface.co` | Only when you run `/vision` and confirm the size |
| 4 | **Language server** | The server's own release source | Only when you install one through `/lsp` and confirm. `OPENCODE_DISABLE_LSP_DOWNLOAD=1` is the default, so nothing is fetched otherwise |
| 5 | **Voice** | The recorder runtime (about 60 MB) and speech model (about 514 MB) | The first time you use voice input, after a consent screen that lists both |
| 6 | **`websearch` tool** | DuckDuckGo by default; Exa or Parallel if you set a key | Whenever the model calls it |
| 7 | **`webfetch` tool** | The URL named in the call | Whenever the model calls it. Page text is capped at 20,000 characters (`LOCALCODE_WEBFETCH_MAX_CHARS`) |
| 8 | **MCP servers** | Wherever you pointed them | Whenever the model calls one of their tools |
| 9 | **Shell commands** | Wherever the command goes | Whenever a `bash` call runs, subject to the permission prompt |

Web search works out of the box with a keyless DuckDuckGo backend. To use a
paid search API instead, set `LOCALCODE_ENABLE_EXA=1` with `EXA_API_KEY`, or
`LOCALCODE_ENABLE_PARALLEL=1` with `PARALLEL_API_KEY`. The key is sent only to
that provider.

The web tools and MCP tools run without a permission prompt. Shell commands
prompt; see [Permissions](/localcode/start-here/permissions).

## The classic interface

`localcode --classic` and the headless `localcode run` use the 0.3 agent loop.
It has a configurable inference endpoint and a connectivity probe that the
default interface does not have. See the 0.3 docs through the version switcher
in the header.
