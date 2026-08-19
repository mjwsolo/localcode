---
title: MCP
description: Connect Model Context Protocol servers to localcode.
---

localcode is an MCP client. Servers you configure expose their tools to the
agent alongside the built-in ones.

## Configure

Servers live in `~/.localcode/mcp.json` under the `mcpServers` key — the same
shape other MCP clients use:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/work"]
    }
  }
}
```

If you set `LOCALCODE_HOME`, the file is read from there instead.

stdio, HTTP and SSE transports are supported, via the official MCP SDK.

## Manage from the TUI

```text
/mcp        # list configured servers, or reload them after editing mcp.json
```

Servers are also connected for headless `localcode run --json`, not just the
TUI.

## Trust

An MCP server is an arbitrary program, and a remote one sees whatever arguments
the model passes to its tools. Configuring a server is a decision to trust it —
it sits outside the local-only boundary described in
[Network Boundary](/localcode/concepts/network-boundary).

:::note[Preview stub]
A worked end-to-end recipe is not on this preview site yet. The repository has
a tested walkthrough in
[`docs/mcp-lsp.md`](https://github.com/mjwsolo/localcode/blob/main/docs/mcp-lsp.md),
which should be migrated here.
:::
