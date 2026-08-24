---
title: MCP
description: Connect Model Context Protocol servers to localcode.
---

localcode is an MCP client. The servers you configure give the agent access to their tools alongside the built-in tools.

## Configure

Add servers to the `mcpServers` key in `~/.localcode/mcp.json`. This uses the same format as other MCP clients:

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

If you set `LOCALCODE_HOME`, localcode reads the file from that location instead.

The official MCP SDK supports stdio, HTTP, and SSE connections.

## Manage from the TUI

```text
/mcp        # list configured servers, or reload them after editing mcp.json
```

