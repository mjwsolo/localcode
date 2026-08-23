---
title: Slash Commands
description: Commands available inside the localcode app.
---

Type `/` in the chat box to open the command list. A line is only a command if its first word is one of the commands below, so a path like `/Users/you/project` is sent to the model as normal text.

Start a line with `!` to run a shell command yourself.

| Command | What it does |
| --- | --- |
| `/status` | Server health, current model, performance settings |
| `/model` | List or switch models, for example `/model qwen` |
| `/permissions` | Turn command approvals on or off |
| `/status` | Show server health, the current model, and where files live |
| `/restart` | Restart the model server. Use it when `/status` shows the server is unreachable |
| `/model` | List models, or switch with `/model <name>` |
| `/delete` | Delete a downloaded model to free disk space. Asks first |
| `/mcp` | List MCP servers, or reload them after editing `mcp.json` |
| `/skills` | List loaded skills and where they came from |
| `/thinking` | Show or set the hidden-reasoning policy: `off` or `auto` |
| `/vision` | Let the model see images |
| `/voice` | Push-to-talk dictation into the input box |
| `/audio` | Read replies aloud with macOS `say` |
| `/sounds` | Turn completion and approval sounds on or off |
| `/search` | Search the conversation, like `Ctrl+F` |
| `/clear` | Clear the conversation |
| `/exit` | Quit localcode |

`/quit` is the same as `/exit`.

`/thinking` has no effect on models without a hidden-reasoning channel; localcode tells you when that is the case.
