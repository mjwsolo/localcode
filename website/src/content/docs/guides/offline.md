---
title: Offline
description: What works with no network, and what doesn't.
---

localcode is local-first, but it is not offline-only. With the default setup, the features you use most do not need a connection after the model is saved on disk. However, some features still need a network.

## Works without a network

- Model inference, as long as `runtime.base_url` points to the local server. A custom endpoint on another host needs a network connection.
- Reading, editing, and writing files
- `grep`, `glob`, `list_files`, code navigation, and symbol inspection
- Shell commands that do not connect to the network
- Syntax checks and local test or lint commands in your repo
- Session resume, `/undo`, and the project event log

## Needs a network

- **The first model download.** This is the only blocking step. Complete it before going offline.
- **Browsing other quantisations** in the model picker. This uses a Hugging Face API call, but cached results still appear.
- **`web_search` and `web_fetch`**, because they use the web.
- **Remote MCP servers**, and any shell command that downloads something.
- **Enabling voice** for the first time, because it downloads speech models.
- **Picking an experimental model** when localcode must clone and build its runner from source.
- **The first semantic-index build**, if `sentence-transformers` is installed. It downloads an embedding model in the background. Without that package, the index uses a local TF-IDF build and needs no network.

localcode also runs a small **connectivity probe**. It makes a TCP connection to `1.1.1.1:443` and caches the result for 30 seconds. It sends no application data. It opens and closes the connection. When offline, the probe simply fails. The model is then told that the machine has no network, so it stops trying to download files.

The current known outbound paths are listed in [Network Boundary](/localcode/concepts/network-boundary). That list may not be complete. Use the two sections above as a general guide to offline use, not as a full guarantee.

## Preparing for an offline session

1. Start localcode once while connected to the network. Wait for the recommended model to finish downloading.
2. Use `/status` to confirm that the server is healthy and the model is loaded.
3. If your project needs dependencies, install them first. An approved `pip install` or `npm install` will fail without a network, like any other network command.

When offline, `web_search` and `web_fetch` calls will return errors. The agent uses the failed tool result as context and continues.

See [Network Boundary](/localcode/concepts/network-boundary) for the current known outbound paths. This list may not be complete.
