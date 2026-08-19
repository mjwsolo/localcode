---
title: CLI
description: Every flag and subcommand localcode accepts.
---

```text
localcode [--profile P] [--model TAG] [--resume SESSION_ID] [-c DIR]
          [--preview-screen SCREEN]
localcode run --goal "..." [options]
localcode unstick
```

Running bare `localcode` starts the TUI. That is the product: first-run setup,
configuration, model management and benchmarking all live inside it. **There is
no `localcode setup` subcommand.**

## Global options

| Flag | Description |
| --- | --- |
| `--profile P` | Gemma 4 profile: `e2b`, `e4b`, `26b-laptop`, `26b-moe`, `31b` |
| `--model TAG` | Explicit local runtime model tag |
| `--resume SESSION_ID` | Resume a previous session; `--resume last` for the most recent. Session IDs are printed on exit |
| `-c`, `--cwd DIR` | Project working directory (defaults to the current directory) |
| `--preview-screen SCREEN` | Visual-test one screen with mock state: `setup`, `mode-picker`, `model-picker`, `chat`. Starts no server and no model |

## `localcode run`

Run a single coding goal headlessly and exit — for scripting, CI and the
benchmark harness. Approvals are forced to full-auto, because there is no human
to answer a prompt.

| Flag | Description |
| --- | --- |
| `--goal TEXT` | **Required.** The task for the agent to perform |
| `--binary PATH` | Path to a `llama-server` binary (e.g. stock llama.cpp on Linux CI; pair with `LOCALCODE_SERVER_FLAVOR=vanilla`) |
| `--timeout N` | Abort after N seconds (`0` = no limit) |
| `--max-rounds N` | Maximum model/tool rounds (`0` = unlimited) |
| `--thinking off\|auto\|on` | Hidden-reasoning policy for this run |
| `--thinking-budget N` | Reasoning-token budget (`0` = model default, negative disables) |
| `--quiet` | Suppress streamed output; print only the final answer |
| `--json` | Emit the event stream as JSON Lines on stdout |

Exit codes: `0` ok · `1` error · `124` timeout · `130` interrupted.

See [Headless](/localcode/guides/headless) and
[JSONL Events](/localcode/reference/jsonl-events).

## `localcode unstick`

Recover from a stuck `llama-server` without rebooting. Runs `memory_pressure`
and `purge`; requires admin rights.

## Environment variables

| Variable | Effect |
| --- | --- |
| `LOCALCODE_AUTONOMY` | `suggest`, `auto_edit` (default) or `full_auto` |
| `LOCALCODE_HOME` | Override `~/.localcode` |
| `LOCALCODE_SERVER_FLAVOR` | `vanilla` to pair with a stock llama.cpp binary |
| `LOCALCODE_ALLOW_DEBUGGER` | `1` to skip the macOS anti-debugger hardening |
