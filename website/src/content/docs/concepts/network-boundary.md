---
title: Network Boundary
description: Exactly what stays on your Mac, what reaches the network, and when.
---

**Your code and prompts stay on your Mac unless you explicitly use a network
tool.**

That sentence is precise on purpose. localcode is not "fully offline" — a few
specific actions reach the network, and this page lists all of them so you can
decide which ones you want.

## What never leaves your Mac

- **Inference.** Generation runs against a `llama-server` process on
  `http://localhost:8081`, started from the binary shipped inside the wheel.
  There is no hosted model, no API key, and no remote fallback.
- **Your files, prompts and edits.** The agent reads and writes your working
  tree directly.
- **Session and event logs.** `<project>/.localcode/events.jsonl` is an
  append-only local audit trail of tool calls, server lifecycle and turn
  boundaries. It is never uploaded.
- **Turn telemetry.** `~/.localcode/telemetry/turns.jsonl` is written for local
  analysis only. Nothing ships it anywhere.

There is no usage ping, no analytics endpoint, and no version check.

## What does reach the network

| Trigger | Destination | When |
| --- | --- | --- |
| Downloading a model | Hugging Face | First launch, and whenever you pick a model you don't have |
| Browsing quantisations in the model picker | Hugging Face repo tree API | Only while browsing; results are cached |
| `web_search` tool | DuckDuckGo (default provider) | Only when the model calls the tool |
| `web_fetch` tool | The URL you or the model names | Only when the model calls the tool |
| Installing a skill from a URL | That URL | Only when you install one |
| MCP servers | Wherever you pointed them | Only for servers you configured in `~/.localcode/mcp.json` |
| Shell commands | Wherever the command goes | Only for `bash` calls you approve |

`web_search` can be pointed at Google, Brave or SerpAPI instead of DuckDuckGo
by putting a key in your config. Doing so sends the query to that provider.

## The parts to be deliberate about

**`web_search` and `web_fetch` send content off the machine.** A search query
is written by the model from your conversation, so treat any use of these tools
as sending that text to a third party.

**MCP servers are arbitrary programs.** A remote MCP server sees whatever
arguments the model passes to its tools. Only configure servers you trust —
see [MCP](/localcode/guides/mcp).

**Approved shell commands can do anything a shell can do.** The permission
prompt on `bash` is where you control that; see
[Permissions](/localcode/start-here/permissions).

## Running with no network at all

Once a model is downloaded, the agent loop, file tools, shell tools and
inference all work without a connection. What breaks is exactly the table
above. See [Offline](/localcode/guides/offline).

## Verifying it yourself

localcode is Apache-2.0, so the boundary is auditable. The outbound call sites
are `bootstrap.py` and `hf_quants.py` (Hugging Face), `tools/web_search.py` and
`tools/web_fetch.py` (the two network tools), `skills.py` (installing a skill
from a URL), and the MCP client. `runtime.py`, `launcher.py` and
`server_manager.py` talk to `localhost` only.

You can also watch what happened after the fact:

```sh
tail -f .localcode/events.jsonl | jq .
```
