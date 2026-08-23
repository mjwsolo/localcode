---
title: Offline
description: What works with no network, and what does not.
---

Once a model is downloaded, localcode works without a network connection.

## Works offline

- Generating code and answers
- Reading, editing and searching files
- Shell commands, tests, builds and lint
- Session resume, `/undo` and the project event log

## Needs a network

- Downloading a model, including switching to one you do not have yet
- Browsing other quantisations in the model picker
- The `web_search` and `web_fetch` tools
- Remote MCP servers
- Turning on voice for the first time, which downloads speech models
- Any shell command that downloads something, such as `pip install`

localcode checks for a connection at the start of a turn by opening and closing a TCP connection to `1.1.1.1:443`. It sends no data. When the check fails, the model is told the machine is offline and stops trying to download things.

## Before going offline

1. Start localcode while connected and let the model download finish.
2. Run `/status` to confirm the model is loaded.
3. Install your project's dependencies first.

Offline, `web_search` and `web_fetch` return errors and the agent carries on without them.

See [Network Boundary](/localcode/concepts/network-boundary) for the complete list of network paths.
