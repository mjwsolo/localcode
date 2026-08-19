---
title: Offline
description: What works with no network, and what doesn't.
---

localcode is local-first, not offline-only. In the default configuration, once
a model is on disk, the parts you use most need no connection — but several
things still do, and pretending otherwise would be a lie.

## Known to work with no network

- Model inference — provided `runtime.base_url` still points at the local
  server. A custom endpoint on another host needs the network like anything
  else.
- Reading, editing and writing files
- `grep`, `glob`, `list_files`, code navigation, symbol inspection
- Shell commands that don't themselves reach out
- Syntax checks and your repo's local test/lint commands
- Session resume, `/undo`, the project event log

## Known to need a network

- **The first model download.** This is the one blocking step; do it before you
  go offline.
- **Browsing other quantisations** in the model picker (a Hugging Face API
  call; cached results still show).
- **`web_search` and `web_fetch`**, by definition.
- **Remote MCP servers**, and any shell command that fetches something.
- **Enabling voice** for the first time, which downloads speech models.
- **Picking an experimental model** whose runner localcode has to clone and
  build from source.
- **The first semantic-index build**, if you have `sentence-transformers`
  installed — it downloads an embedding model in the background. Without that
  package the index falls back to a local TF-IDF build and needs nothing.

localcode also runs a small **connectivity probe** — a TCP connect to
`1.1.1.1:443`, cached for 30 seconds. No application payload is sent; the
connection is opened and closed. Offline it simply fails, and the model is told
the machine has no network so it stops attempting downloads.

The known outbound paths in the current implementation are listed on
[Network Boundary](/localcode/concepts/network-boundary). That list is not
proven complete, so treat the two sections above as the shape of the offline
story rather than an exhaustive guarantee.

## Preparing for an offline session

1. Launch localcode once on a connection and let the recommended model finish
   downloading.
2. Confirm with `/status` that the server is healthy and the model is loaded.
3. If your project installs dependencies, install them first — an approved
   `pip install` or `npm install` will fail without a network like any other
   command.

Offline, expect `web_search` / `web_fetch` calls to come back as errors. The
agent treats a failed tool result as context and continues.

See [Network Boundary](/localcode/concepts/network-boundary) for the complete
inventory.
