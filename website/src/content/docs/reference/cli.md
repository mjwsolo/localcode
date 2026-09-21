---
title: CLI
description: Every flag and subcommand localcode accepts.
---

```text
localcode [-c DIR] [--model TAG] [--classic]
localcode --version
localcode run --goal "..." [options]
localcode unstick
```

Run `localcode` by itself to open the interface. Setup, the model picker, and sessions all live inside it. There is no `localcode setup` subcommand and no benchmark subcommand.

## Options

| Flag | Description |
| --- | --- |
| `-c`, `--cwd DIR` | Project directory. The default is the current directory |
| `--model TAG` | Start with an already-downloaded model alias instead of the picker |
| `--classic` | Open the previous 0.3 interface. `LOCALCODE_FRONTEND=classic` does the same |
| `--version` | Print the version and exit |

`--profile`, `--resume`, and `--preview-screen` belong to the classic interface and imply `--classic`.

One launcher runs per user. A second `localcode` is refused with a hint to use `/models` in the running one, or to exit it first.

## `localcode run`

Run one coding goal without the interface, then exit. Use this for scripts, CI, and evaluation. This is the headless agent from 0.3 and is unchanged. Approvals always use full-auto because no person is available to answer prompts; writes outside the project directory are rejected.

| Flag | Description |
| --- | --- |
| `--goal TEXT` | **Required.** The task the agent must complete |
| `--binary PATH` | Path to a `llama-server` binary. For example, use stock llama.cpp on Linux CI with `LOCALCODE_SERVER_FLAVOR=vanilla` |
| `--timeout N` | Stop after N seconds (`0` = no limit) |
| `--max-rounds N` | Maximum number of model/tool rounds (`0` = unlimited) |
| `--thinking off\|auto\|on` | Hidden-reasoning setting for this run |
| `--thinking-budget N` | Reasoning-token limit (`0` = model default, negative disables) |
| `--quiet` | Hide streamed output and print only the final answer |
| `--json` | Write the event stream to stdout as JSON Lines |

Exit codes: `0` ok · `1` error · `124` timeout · `130` interrupted.

See [JSONL Events](/localcode/reference/jsonl-events).

## `localcode unstick`

Recovers from a stuck `llama-server` without a reboot. It runs `memory_pressure` and `purge`, which need admin rights.

## Environment variables

| Variable | Effect |
| --- | --- |
| `LOCALCODE_FRONTEND` | `ui` (default) or `classic` |
| `LOCALCODE_MODELS_DIR` | Where GGUFs live. Default `~/.local/share/localcode/models` |
| `LOCALCODE_AGENT_RUN_DIR` | Supervisor logs and the voice runtime. Default `~/.local/share/localcode-agent/run` |
| `LOCALCODE_MIC` | Microphone device for voice input |
| `LOCALCODE_ENABLE_EXA=1` + `EXA_API_KEY` | Use Exa for web search instead of the keyless default |
| `LOCALCODE_ENABLE_PARALLEL=1` + `PARALLEL_API_KEY` | Use Parallel for web search instead of the keyless default |
| `LOCALCODE_WEBFETCH_MAX_CHARS` | Cap on fetched page text. Default `20000` |
| `OPENCODE_DISABLE_LSP_DOWNLOAD` | Default `1`: language servers are never downloaded without `/lsp` |
| `LOCALCODE_UI_BIN` | Developer override: path to the interface binary |
| `LOCALCODE_LLAMA_SERVER` | Developer override: path to a `llama-server` binary |

The full list, with the classic-only variables, is in [Configuration](/localcode/reference/configuration).
