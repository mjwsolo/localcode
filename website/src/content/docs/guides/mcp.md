---
title: MCP
description: Connect Model Context Protocol servers to localcode.
---

localcode is an MCP client. The servers you add give the agent access to their tools alongside the built-in tools.

## Add a server

Add servers to the `mcpServers` key in `~/.localcode/mcp.json`, then run `/mcp reload` in the TUI. This is the same file format other MCP clients use:

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

A remote server that needs a token takes it as a header:

```json
{
  "mcpServers": {
    "github": {
      "transport": "http",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```

Set `LOCALCODE_HOME` to read the file from another location. The MCP SDK supports stdio, HTTP, and SSE connections.

## From the TUI

```text
/mcp          # list configured servers and their tools
/mcp reload   # re-read mcp.json and reconnect after an edit
```
