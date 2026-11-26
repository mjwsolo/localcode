---
title: MCP
description: Give the agent tools from Model Context Protocol servers, such as mail, calendar or GitHub.
---

localcode is an MCP client. A server you add gives the agent that server's tools next to the built-in ones: read a mail thread, put a hold in a calendar, open an issue. Everything runs from the terminal you are already in; the model stays local, and only the server you chose talks to its service.

## Add a server

Same form as Claude Code: the command goes after `--`.

```
localcode mcp add <name> -- <command> [args...]    # a local server
localcode mcp add <name> --url <url>               # a remote server
localcode mcp add --global <name> -- <command>     # keep it for every project
localcode mcp remove <name>                        # add --global for a user-config server
```

Local servers are programs that run on your Mac; remote servers are URLs, and the ones that use OAuth sign in with:

```
localcode mcp auth <name>
```

Then check it:

```
localcode mcp list
localcode mcp debug <name>     # connect and print the tools it exposes
```

The server is in your config from then on. In a session, `/mcps` lists the configured servers and toggles each one on or off.

## Import from other agents

If you already use MCP servers in Claude Code, Codex or Cursor, copy them over:

```
localcode mcp import                  # asks which ones
localcode mcp import --dry-run        # show what would happen, write nothing
localcode mcp import --from codex     # only one source
```

It reads only user-level files, and only when you run this command: `~/.claude.json` (its `mcpServers` entry), `~/.codex/config.toml` and `~/.cursor/mcp.json`. Project `.mcp.json` files are never read. You pick the servers by number, then see a summary that shows each local server's exact command line and the names (never the values) of its environment variables and headers. The values are copied into your localcode config, so those secrets then exist in two places.

Nothing runs at import. Every imported server is written with `"enabled": false`; turn it on in `/mcps` once you have checked its command. Your config is backed up to `localcode.json.bak-<timestamp>` before it is changed, a server whose name is already taken by a different one is skipped with a suggested new name, and running the import again changes nothing. The same command also offers your Claude Code skills; see [Skills](/localcode/guides/skills-and-hooks).

## Manage servers in a session

`/mcps` is a plain list: each server's name, whether it is on or off, and its status (connected, failed, or needs sign-in). Space turns the highlighted server on or off, and the choice is saved to your config. Enter opens sign-in for a server that needs it, or shows the error for one that failed. Adding and removing servers happens in the terminal with `localcode mcp add` and `localcode mcp remove`; the list shows that reminder at the bottom.

## Project or user scope

A server kept in your user config (`~/.config/localcode-agent/localcode.json`) is available in every project. A server kept in the project's `localcode.json` is part of that project's configuration, and project configuration can run code, so it loads only when you launch with:

```
localcode --trust-project
```

`localcode mcp add` tells you this when it writes to the project scope, and `localcode mcp list` names project servers it is hiding until you pass `--trust-project`. Review a project's server commands before trusting it.

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
