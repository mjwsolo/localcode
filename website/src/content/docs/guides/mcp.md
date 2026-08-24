---
title: MCP
description: Connect Model Context Protocol servers to localcode.
---

localcode is an MCP client. The servers you add give the agent access to their tools alongside the built-in tools.

## Add a server

Type `/mcp` in the TUI and press `a`. Paste a URL for a remote server, or type a command for a local one:

- Remote: `https://api.githubcopilot.com/mcp/`
- Local: `npx -y @modelcontextprotocol/server-filesystem /Users/you/work`

localcode writes the server to `~/.localcode/mcp.json` and connects it. Give it a name, or leave that blank and localcode derives one from the URL or command.

## Manage from the TUI

```text
/mcp     # list servers and their tools
         # a add · r reload · d disconnect
```

## The config file

`/mcp` writes `~/.localcode/mcp.json`, the same format other MCP clients use. Edit it directly if you prefer, then press `r` in `/mcp` to reload:

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
