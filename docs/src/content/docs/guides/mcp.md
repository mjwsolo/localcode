---
title: MCP
description: Connect Model Context Protocol servers to localcode.
---

localcode is an MCP client. The servers you add give the agent access to their tools alongside the built-in tools.

## Add a server

Declare servers under the `mcp` key of `localcode.json` in the project root. A local server is a command to run; a remote server is a URL:

```json
{
  "$schema": "https://localcode.dev/schema/config.json",
  "mcp": {
    "filesystem": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "github": {
      "type": "remote",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```

Add `"enabled": false` to a server to keep it configured but off. Local servers can take an `"environment"` object for variables the command needs. The file follows the opencode config schema; see [Configuration](/localcode/reference/configuration).

## From the interface

`/mcps` lists the configured servers and toggles each one on or off for the session. The tools of every enabled server appear to the model next to the built-in tools.

MCP tools run without a permission prompt when the model calls them. A remote server is a network path; see [Network Boundary](/localcode/concepts/network-boundary).
