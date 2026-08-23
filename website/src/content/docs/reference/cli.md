---
title: CLI
description: Every flag and subcommand localcode accepts.
---

```text
localcode [--model TAG] [--resume SESSION_ID] [-c DIR]
localcode run --goal "..." [options]
localcode unstick
```

`localcode` on its own opens the app. `lc` is a shorter alias.

## Options

| Flag | Description |
| --- | --- |
| `--model TAG` | Start with a specific model |
| `--resume SESSION_ID` | Continue an earlier session. `--resume last` picks the most recent. Session IDs are printed when you exit |
| `-c`, `--cwd DIR` | Project directory. Defaults to the current directory |
| `--profile P` | Gemma 4 profile: `e2b`, `e4b`, `26b-laptop`, `26b-moe`, `31b` |
| `--preview-screen SCREEN` | Open one screen with mock data, without starting a model: `setup`, `mode-picker`, `model-picker`, `chat` |

## `localcode run`

Run one task without the UI, then exit. Useful for scripts and CI. Approvals are always off, so it never waits for you.

| Flag | Description |
| --- | --- |
| `--goal TEXT` | Required. The task to do |
| `--timeout N` | Stop after N seconds. `0` means no limit |
| `--max-rounds N` | Maximum model/tool rounds. `0` means unlimited |
| `--thinking off\|auto\|on` | Hidden reasoning for this run |
| `--thinking-budget N` | Reasoning-token limit. `0` uses the model default; negative turns it off |
| `--quiet` | Print only the final answer |
| `--json` | Write the event stream to stdout as JSON Lines. See [JSONL Events](/localcode/reference/jsonl-events) |
| `--binary PATH` | Use a different `llama-server` binary |

Exit codes: `0` done, `1` error or incomplete, `124` timeout, `130` interrupted.

## `localcode unstick`

Recover a stuck model server without rebooting. Needs admin rights.

## Environment variables

| Variable | Effect |
| --- | --- |
| `LOCALCODE_AUTONOMY` | `suggest`, `auto_edit` (default) or `full_auto`. See [Permissions](/localcode/start-here/permissions) |
| `LOCALCODE_MODEL` | Model tag, same as `--model` |
| `LOCALCODE_BASE_URL` | Send prompts to a different server. See [Network Boundary](/localcode/concepts/network-boundary) |
| `LOCALCODE_HOME` | Use a directory other than `~/.localcode` |
| `LOCALCODE_PORT` | Port for the local model server |
| `LOCALCODE_TELEMETRY` | `0` leaves UI turn summaries out of the local event log |
| `LOCALCODE_TRUST_REMOTE_CODE` | `1` lets the optional code-search embedding model run Python downloaded from its Hugging Face repo. Off by default |
