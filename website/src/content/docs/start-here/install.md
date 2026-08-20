---
title: Install
description: Requirements, installation, and what happens on first launch.
---

## Requirements

| | |
| --- | --- |
| Machine | Mac with Apple Silicon |
| Unified memory | At least 16 GB |
| Python | 3.10 or newer |
| Disk | Space for one model — the smallest recommended GGUF is about 7.4 GB |

Apple Silicon is the supported platform. Metal-accelerated inference works only on Mac. localcode also installs and runs on Linux in CI for development, but Linux is not the product platform.

## Install

```sh
pip install -U localcode
```

You can also use `uv pip install -U localcode`. The wheel includes a prebuilt `llama-server` binary. This is a llama.cpp fork with TurboQuant KV-cache compression. The default installation does not need a compiler.

## Run

```sh
cd your-project
localcode
```

That is the complete command. The plain `localcode` command is the product. First-run setup, model selection, configuration, and model management are all in the TUI. There is no `localcode setup` subcommand.

There are only two other entry points:

```sh
localcode run --goal "..."   # headless, one goal, then exit
localcode unstick            # recover a wedged model server
```

## What happens on first launch

1. localcode checks your Mac's unified memory and recommends a model. See [Choose a Model](/localcode/start-here/choose-a-model).
2. It downloads the model's GGUF from Hugging Face. This is the only step that needs the network. You only need to do it once for each model.
3. By default, it starts the included `llama-server` at `http://localhost:8081` and connects the agent to it. If you set `runtime.base_url` or `LOCALCODE_BASE_URL` to another address, the agent connects to that address instead.

After that, generation happens wherever `runtime.base_url` points. By default, this is the server on your Mac. If you change it or `LOCALCODE_BASE_URL`, your prompts and code context go to the address you set. The value is not checked. See [Network Boundary](/localcode/concepts/network-boundary) to learn what stays on your machine and what leaves it.

## Where localcode keeps things

| Path | What |
| --- | --- |
| `~/.localcode/config.toml` | Global settings |
| `~/.localcode/mcp.json` | MCP server settings |
| `~/.localcode/skills/` | Global skills |
| `<project>/.localcode/` | Project state, including `events.jsonl` |
| `<project>/.localcode/config.toml` | Project settings applied over global settings |

You can safely add everything under `<project>/.localcode/` to `.gitignore`.

## If something goes wrong

- Every error shown to users has a stable `Eccc` code. See [Error Codes](/localcode/reference/error-codes).
- If the model server gets stuck because a `llama-server` from an earlier session is still running, use `localcode unstick`. It runs `memory_pressure` and `purge` and needs admin rights.
- Detailed information about the latest project error is saved in `<project>/.localcode/last_error.log`.

:::caution[Alpha software]
localcode is still being actively developed. Expect some problems and breaking changes between versions.
:::

## Next

[Get started →](/localcode/start-here/first-change)
