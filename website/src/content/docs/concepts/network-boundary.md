---
title: Network Boundary
description: The network paths localcode can use, what triggers them, and what stays on your Mac by default.
---

**With the default setup, inference runs on your Mac. It needs no API key or
model provider.** A few features do use the network, and one setting can move
inference off your Mac. This page lists each one.

## What stays on your Mac by default

- **Inference.** Generation sends requests to `<base_url>/v1/chat/completions`.
  By default that is a `llama-server` process at `http://localhost:8081`, using
  the binary included in the wheel.
- **Your files, prompts and edits.** The agent reads and writes your working
  tree directly.
- **Session and event logs.** `<project>/.localcode/events.jsonl` is a local,
  append-only record of tool calls, turn boundaries and server lifecycle. It is
  never uploaded. Set `LOCALCODE_TELEMETRY=0` to turn off the UI turn-trace
  records inside it.

There is no analytics endpoint, usage reporting or version check.

## Inference endpoint

`runtime.base_url` in `~/.localcode/config.toml`, or the `LOCALCODE_BASE_URL`
environment variable, controls where chat completions are sent. **It accepts
any URL and is not limited to localhost.** If you point it at a remote server,
localcode sends every prompt it builds to that server: your message, the file
contents gathered for context, tool results and the model's replies. Check the
current value with `/status`.

## Where localcode uses the network

| # | What | Where it goes | When |
| --- | --- | --- | --- |
| 1 | **Connectivity probe** | TCP connect to `1.1.1.1:443` | Automatic, at most once per turn (cached 30 s). No payload - it opens and closes the connection to check for internet before a download |
| 2 | **Model download** | `huggingface.co` | On first launch, and whenever you choose a model you do not have yet |
| 3 | **Quant browsing** | Hugging Face repo tree API | Only when you browse other quantisations in the model picker. Cached |
| 4 | **`llama-server` fallback binary** | `github.com/mjwsolo/localcode` Releases | Only when the included binary cannot be used. Refused if TLS verification fails |
| 5 | **Voice model** (optional `voice` extra) | `huggingface.co/ggerganov/whisper.cpp` | The first time you enable voice input |
| 6 | **Voice output voices** (optional `voice` extra) | `huggingface.co/rhasspy/piper-voices` | The first time a speech voice is used |
| 7 | **`web_search` tool** | DuckDuckGo | Whenever the model calls it |
| 8 | **`web_fetch` tool** | The URL named in the call | Whenever the model calls it |
| 9 | **Skill install from a URL** | That URL | Only when you install one that way |
| 10 | **MCP servers** | Wherever you pointed them | Whenever the model calls one of their tools |
| 11 | **Shell commands** | Wherever the command goes | Whenever a `bash` or `background_process` call runs |
| 12 | **Custom inference endpoint** | Whatever `base_url` names | Every turn, if you changed it from the default |

The network tools (`web_search`, `web_fetch`, and MCP tools) run without a
confirmation prompt at every autonomy level. `localcode run` (headless) forces
`full_auto`, so a headless run can make network requests without asking. See
[Permissions](/localcode/start-here/permissions) for what does and does not
prompt.
