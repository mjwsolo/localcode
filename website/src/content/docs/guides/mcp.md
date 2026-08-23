---
title: MCP
description: Connect Model Context Protocol servers to localcode.
---

localcode is an MCP client. Tools from the servers you configure appear alongside the built-in tools.

## Configure

Add servers under `mcpServers` in `~/.localcode/mcp.json`. The format is the same as other MCP clients:

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

stdio, HTTP and SSE servers are supported. If you set `LOCALCODE_HOME`, the file lives there instead.

## Manage

```text
/mcp        # list configured servers, or reload after editing mcp.json
```

Servers also connect for headless `localcode run`.

## Trust

An MCP server is a program you are choosing to run, and a remote server sees every argument the model sends it. MCP tools run without an approval prompt. Only add servers you trust. See [Network Boundary](/localcode/concepts/network-boundary).
