---
title: CLI
description: Every flag and subcommand localcode accepts.
---

```text
localcode [--profile P] [--model TAG] [--resume SESSION_ID] [-c DIR]
          [--preview-screen SCREEN]
localcode run --goal "..." [options]
```

Run `localcode` by itself to start the TUI. The TUI is the product. It includes first-run setup, configuration, and model management. **There is no `localcode setup` subcommand.** There is also no benchmark subcommand or screen. The speeds in the model picker are [calculated estimates](/localcode/start-here/choose-a-model#the-toks-numbers-in-the-model-picker), not measurements.

## Global options

| Flag | Description |
| --- | --- |
| `--profile P` | Gemma 4 profile: `e2b`, `e4b`, `26b-laptop`, `26b-moe`, `31b` |
| `--model TAG` | Exact local runtime model tag |
| `--resume SESSION_ID` | Continue an earlier session. Use `--resume last` for the most recent session. Session IDs appear when you exit |
| `-c`, `--cwd DIR` | Project working directory. The default is the current directory |
| `--preview-screen SCREEN` | Test one screen visually with mock data: `setup`, `mode-picker`, `model-picker`, `chat`. This does not start a server or model |

## `localcode run`

Run one coding goal without the TUI, then exit. Use this for scripts, CI, and evaluation. Approvals always use full-auto because no person is available to answer prompts.

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

## Environment variables

| Variable | Effect |
| --- | --- |
| `LOCALCODE_AUTONOMY` | `suggest`, `auto_edit` (default), or `full_auto` |
| `LOCALCODE_HOME` | Use a location other than `~/.localcode` |
| `LOCALCODE_SERVER_FLAVOR` | Use `vanilla` with a stock llama.cpp binary |
| `LOCALCODE_ALLOW_DEBUGGER` | Set to `1` to skip macOS anti-debugger hardening |
