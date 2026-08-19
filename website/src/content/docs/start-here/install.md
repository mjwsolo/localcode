---
title: Install
description: Requirements, installation, and what happens on first launch.
---

## Requirements

| | |
| --- | --- |
| Machine | Mac with Apple Silicon |
| Unified memory | 16 GB minimum |
| Python | 3.10 or newer |
| Disk | Room for one model — the smallest recommended GGUF is about 7.4 GB |

Apple Silicon is the supported target: the Metal-accelerated inference path is
Mac-only. localcode installs and runs on Linux in CI for development, but that
path is not the product.

## Install

```sh
pip install -U localcode
```

`uv pip install -U localcode` works too. The wheel ships a prebuilt
`llama-server` binary (a llama.cpp fork with TurboQuant KV-cache
compression), so there is no compiler step for the default path.

## Run

```sh
cd your-project
localcode
```

That is the whole command. The bare `localcode` invocation is the product —
first-run setup, model choice, configuration and model management all live
inside the TUI. There is no `localcode setup` subcommand.

The only other entry points are:

```sh
localcode run --goal "..."   # headless, one goal, then exit
localcode unstick            # recover a wedged model server
```

## What happens on first launch

1. localcode reads your Mac's unified memory and recommends a model — see
   [Choose a Model](/localcode/start-here/choose-a-model).
2. It downloads that model's GGUF from Hugging Face. This is the one step
   that needs the network, and it is a one-time cost per model.
3. It starts the bundled `llama-server` on `http://localhost:8081` and points
   the agent at it.

From then on, generation happens wherever `runtime.base_url` points — by
default, that server on your Mac. Changing it (or `LOCALCODE_BASE_URL`) sends
your prompts and code context to whatever you name instead; the value is not
validated. See [Network Boundary](/localcode/concepts/network-boundary) for
what does and does not leave the machine.

## Where localcode keeps things

| Path | What |
| --- | --- |
| `~/.localcode/config.toml` | Global configuration |
| `~/.localcode/mcp.json` | MCP server definitions |
| `~/.localcode/skills/` | Global skills |
| `<project>/.localcode/` | Per-project state, including `events.jsonl` |
| `<project>/.localcode/config.toml` | Per-project config, layered over global |

Everything under `<project>/.localcode/` is safe to add to `.gitignore`.

## If something goes wrong

- Every user-facing error has a stable `Eccc` code — see
  [Error Codes](/localcode/reference/error-codes).
- If the model server wedges (a stuck `llama-server` from a previous session),
  run `localcode unstick`. It runs `memory_pressure` and `purge` and needs
  admin rights.
- Verbose detail for the most recent error in a project is written to
  `<project>/.localcode/last_error.log`.

:::caution[Alpha software]
localcode is under active development. Expect rough edges and breaking changes
between versions.
:::

## Next

[Make your first change →](/localcode/start-here/first-change)
