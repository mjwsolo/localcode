---
title: Install
description: Requirements, installation, and what happens on first launch.
---

## Requirements

| | |
| --- | --- |
| Machine | Mac with Apple Silicon |
| Unified memory | 16 GB or more |
| Python | 3.10 or newer |
| Disk | Room for one model. The smallest is about 7.4 GB |

## Install

```sh
pip install -U localcode
```

`uv pip install -U localcode` also works. The inference server ships inside the package, so you do not need a compiler or Xcode.

## Run

```sh
cd your-project
localcode
```

That is the whole command. Setup, model selection and configuration all happen inside the app. Two other commands exist:

```sh
localcode run --goal "..."   # run one task without the UI, then exit
localcode unstick            # recover a stuck model server
```

## First launch

1. localcode checks how much memory your Mac has and recommends a model. You pick one. See [Choose a Model](/localcode/start-here/choose-a-model).
2. It downloads the model weights from Hugging Face. This happens once per model.
3. It starts the model server on your Mac at `http://localhost:8081` and opens the chat screen.

After that, everything runs on your Mac. See [Network Boundary](/localcode/concepts/network-boundary) for the few features that use the network.

## Where localcode keeps things

| Path | What |
| --- | --- |
| `~/.localcode/config.toml` | Settings |
| `~/.localcode/mcp.json` | MCP servers |
| `~/.localcode/skills/` | Skills |
| `~/.local/share/localcode/models/` | Downloaded models |
| `<project>/.localcode/` | Per-project state and the event log |

Add `<project>/.localcode/` to your `.gitignore`.

## If something goes wrong

- Every error has a code like `E1010`. See [Error Codes](/localcode/reference/error-codes).
- If the model server stops responding, run `/restart` in the app, or `localcode unstick` from a terminal (it needs admin rights).
- Details of the last error are in `<project>/.localcode/last_error.log`.

:::caution[Alpha software]
Expect rough edges and breaking changes between versions.
:::

## Next

[Get started](/localcode/start-here/first-change)
