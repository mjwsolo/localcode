---
title: Headless
description: Run one goal with no TUI — for scripts, CI, and editors.
---

```sh
localcode run --goal "add a regression test for parse_config"
```

`run` executes a single goal through the same agent loop the TUI drives, then
exits. It is the entry point for scripting, CI and the benchmark harness.

## Approvals are forced to full-auto

There is no human present to answer a prompt, so headless runs auto-approve
tools. The safety layer still blocks the operations it always blocks — see
[Permissions](/localcode/start-here/permissions). Run headless against a repo
you are willing to have modified, ideally on a branch.

## Options you will actually use

```sh
localcode run --goal "..." \
  --timeout 900 \
  --max-rounds 40 \
  --thinking off \
  --quiet
```

- `--timeout N` — abort after N seconds (`0` = no limit)
- `--max-rounds N` — cap model/tool rounds (`0` = unlimited)
- `--thinking off|auto|on` and `--thinking-budget N`
- `--quiet` — final answer only
- `--json` — machine-readable event stream

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Completed |
| `1` | Error |
| `124` | Timed out |
| `130` | Interrupted |

## Driving it from a program

```sh
localcode run --goal "..." --json | tee run.jsonl | tail -1 | jq .status
```

MCP servers configured in `~/.localcode/mcp.json` are connected for
`run --json` too, not only in the TUI. The stream format is documented in
[JSONL Events](/localcode/reference/jsonl-events).

## Running against a non-default binary

For CI on Linux, pair a stock llama.cpp build with the vanilla server flavour:

```sh
LOCALCODE_SERVER_FLAVOR=vanilla localcode run --goal "..." --binary /path/to/llama-server
```

Apple Silicon remains the supported target; this path exists so the agent loop
can be exercised in CI.
