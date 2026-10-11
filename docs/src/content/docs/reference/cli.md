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
| `--trust-project` | Enable reviewed project configuration, plugins and MCP startup commands for this launch |
| `--resume SESSION_ID` | Resume that session. If no directory is given, LocalCode opens the project saved with the session. Use `--resume last` for the latest session in the current project |
| `--version` | Print the version and exit |

A second `localcode` opens another interface session attached to the running model server. The windows share that model; closing the second window leaves the first session running. Use `/models` to switch the shared model.

`localcode api` reports the running model's loopback endpoint. The API key is only printed with `--show-key`; see [Local API](/localcode/guides/local-api).

`localcode decide` evaluates a typed question using the running OpenJev model and prints JSON. Supply `--state TEXT` or `--state-file PATH`, and optionally `--image PATH`. Repeat `--option` for choice labels or ordered score levels. See [Chat and Decisions](/localcode/guides/decisions).

