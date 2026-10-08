---
title: CLI
description: Every flag localcode accepts.
---

```text
localcode [DIR] [--model TAG] [--resume SESSION_ID]
localcode --version
localcode api [--json] [--show-key]
localcode [DIR] --mode chat|decisions
localcode decide --question TEXT [--type yes-no|choice|score] [--option LABEL ...]
```

Run `localcode` by itself to open the interface. Setup, the model picker, and sessions all live inside it. There is no `localcode setup` subcommand and no benchmark subcommand.

## Options

| Flag | Description |
| --- | --- |
| `DIR` | Project directory. The default is the current directory |
| `--model TAG` | Start with an already-downloaded model alias instead of the picker |
| `--mode chat\|decisions` | Open Chat (default) or the typed decision screen |
| `-s SESSION_ID`, `--session SESSION_ID`, `--resume SESSION_ID` | Resume that session. If no directory is given, LocalCode opens the project saved with the session. Use `--resume last` for the latest session in the current project |
| `--version` | Print the version and exit |

A second `localcode` opens another interface session attached to the running model server. The windows share that model; closing the second window leaves the first session running. Use `/models` to switch the shared model.

`localcode api` reports the running model's loopback endpoint. The API key is only printed with `--show-key`; see [Local API](/localcode/guides/local-api).

`localcode decide` evaluates a typed question using the running OpenJev model and prints JSON. Supply `--state TEXT` or `--state-file PATH`, and optionally `--image PATH`. Repeat `--option` for choice labels or ordered score levels. See [Chat and Decisions](/localcode/guides/decisions).

## Environment variables

| Variable | Effect |
| --- | --- |
| `LOCALCODE_MODEL_DIR` | Where GGUFs live. Default `~/.local/share/localcode/models` |
| `LOCALCODE_MODELS_DIR` | Older name for the models directory override; `LOCALCODE_MODEL_DIR` takes precedence |
| `LOCALCODE_AGENT_RUN_DIR` | Supervisor logs and the voice runtime. Default `~/.local/share/localcode-agent/run` |
| `LOCALCODE_MIC` | Microphone device for voice input |
| `LOCALCODE_ENABLE_EXA=1` + `EXA_API_KEY` | Use Exa for web search instead of the keyless default |
| `LOCALCODE_ENABLE_PARALLEL=1` + `PARALLEL_API_KEY` | Use Parallel for web search instead of the keyless default |
| `LOCALCODE_WEBFETCH_MAX_CHARS` | Cap on fetched page text. Default `20000` |
| `OPENCODE_DISABLE_LSP_DOWNLOAD` | Default `1`: language servers are never downloaded without `/lsp` |
| `LOCALCODE_AUTO_CHECK=1` | Let the plugin run the project's own build or test command after a change, without asking. Off by default: that command is code from the opened repository |
| `LOCALCODE_UI_BIN` | Developer override: path to the interface binary |
| `LOCALCODE_LLAMA_SERVER` | Developer override: path to a `llama-server` binary |

The full list is in [Configuration](/localcode/reference/configuration).
