---
title: Offline
description: What works with no network, and what doesn't.
---

localcode is local-first. After a model is on disk, the things you use most work
with no connection. A few features still need the network, and each one is
something you choose.

## Works offline

- Model inference. The interface only ever talks to the local `llama-server`.
- Reading, editing and writing files.
- `grep`, `glob`, and shell commands that do not reach the network.
- The project's own typecheck, test and lint commands, which the plugin runs
  after edits.
- Sessions, `/undo`, `/diff`, `/review`, and the model picker for models
  already on disk.
- Voice input and vision, once their downloads are on disk.

## Needs the network

- Downloading a model. The picker still opens offline and lists what is on
  disk; the `Available to download` section is empty or fails to load.
- `/vision`, the first time, to fetch a model's image projector.
- `/lsp`, to install a language server. Starting one already installed works
  offline.
- Voice, the first time, to fetch the recorder runtime and speech model.
- `websearch` and `webfetch`.
- Remote MCP servers, and any shell command that downloads something.

## Preparing for an offline session

1. Start localcode once while connected. Pick the starred model in the picker
   and let the download finish.
2. If you want images or voice, run `/vision` and try voice once so their
   downloads complete.
3. Run `/status` to confirm the model is loaded.
4. Install any project dependencies first - a `pip install` or `npm install`
   fails offline like any other network command.

When offline, `websearch` and `webfetch` return errors. The agent uses the
failed result as context and keeps going. See
[Network Boundary](/localcode/concepts/network-boundary) for every path that
uses the network.
