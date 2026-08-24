---
title: Offline
description: What works with no network, and what doesn't.
---

localcode is local-first. After a model is on disk, the things you use most work
with no connection. A few features still need the network.

## Works offline

- Model inference, while `runtime.base_url` points at the local server.
- Reading, editing and writing files.
- `grep`, `glob`, `list_files`, code navigation and symbol inspection.
- Shell commands that do not reach the network.
- Syntax checks and your repo's own test and lint commands.
- Session resume and the project event log.

## Needs the network

- The first model download. This is the one blocking step - do it before going
  offline.
- Browsing other quantisations in the model picker (cached results still show).
- `web_search` and `web_fetch`.
- Remote MCP servers, and any shell command that downloads something.
- Enabling voice for the first time (it downloads speech models).

## Preparing for an offline session

1. Start localcode once while connected and let the recommended model finish
   downloading.
2. Run `/status` to confirm the server is healthy and the model is loaded.
3. Install any project dependencies first - a `pip install` or `npm install`
   fails offline like any other network command.

When offline, `web_search` and `web_fetch` return errors. The agent uses the
failed result as context and keeps going. See
[Network Boundary](/localcode/concepts/network-boundary) for every path that
uses the network.
