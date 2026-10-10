---
title: MCP
description: Give the agent tools from Model Context Protocol servers, such as mail, calendar or GitHub.
---

localcode is an MCP client. A server you add gives the agent that server's tools next to the built-in ones: read a mail thread, put a hold in a calendar, open an issue. Everything runs from the terminal you are already in; the model stays local, and only the server you chose talks to its service.

## Add a server

```
localcode mcp add
```

It asks four things: where to keep it (this project, or your user config for every project), a name, local or remote, and the command or URL. Local servers are programs that run on your Mac; remote servers are URLs, and the ones that use OAuth sign in with:

```
localcode mcp auth <name>
```

Then check it:

```
localcode mcp list
localcode mcp debug <name>     # connect and print the tools it exposes
```

The server is in your config from then on. In a session, `/mcps` lists the configured servers and toggles each one on or off.

## Project or user scope

A server kept in your user config (`~/.config/localcode-agent/localcode.json`) is available in every project. A server kept in the project's `localcode.json` is part of that project's configuration, and project configuration can run code, so it loads only when you launch with:

```
localcode --trust-project
```

`localcode mcp add` tells you this when you choose the project scope, and `localcode mcp list` names project servers it is hiding until you pass `--trust-project`. Review a project's server commands before trusting it.

## Mail, calendar and other accounts

Mail and calendar servers act on your behalf: they can send, delete and book. Two settings keep that under your control.

**Ask before acting.** By default a tool from a server runs when the model calls it. Put the server's tools behind a prompt in your config, by name or pattern:

```json
{
  "permission": {
    "mail_send*": "ask",
    "mail_delete*": "ask",
    "calendar_create*": "ask"
  }
}
```

Tool names are the server name joined to the tool, so `localcode mcp debug <name>` shows the exact names to use. `"ask"` prompts in the interface, `"deny"` removes the tool, `"allow"` is the default.

**Expose only what you need.** Every enabled tool is described to the model on every request, and local models do worse as that list grows. A mail server can expose dozens of tools; deny the ones you will not use.

## The config, if you want to see it

`localcode mcp add` writes this shape; you can edit it directly too:

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

Add `"enabled": false` to keep a server configured but off. Local servers take an `"environment"` object for variables the command needs. The file follows the opencode config schema; see [Configuration](/localcode/reference/configuration).

A remote server is a network path; see [Network Boundary](/localcode/concepts/network-boundary).
