---
title: Slash Commands
description: Commands available inside the localcode TUI.
---

Type `/` in the chat box to open the command palette. Text that starts with `/` is only a command if the first word is a known command. A path like `/Users/you/project` is sent to the model as a normal message.

Start a line with `!` to run a shell command, for example `!git status`. The output appears in the chat log and the model is not involved.

| Command | What it does |
| --- | --- |
| `/permissions` | Turn command approvals on or off |
| `/status` | Show the server health, current model, and performance settings |
| `/restart` | Restart the model server when `/status` shows "unreachable" |
| `/mcp` | List MCP servers and their tools, add one, or reload after editing `~/.localcode/mcp.json` |
| `/skills` | List loaded skills and their sources |
| `/model` | List or switch models, for example `/model qwen` |
| `/delete` | Delete a downloaded model to free disk space after asking first |
| `/thinking` | Show or set the hidden-reasoning policy to `off` or `auto` |
| `/sounds` | Turn completion and approval sounds on or off |
| `/voice` | Turn voice mode on or off for push-to-talk dictation in the input box |
| `/audio` | Turn audio output on or off so macOS `say` can read replies aloud |
| `/vision` | Turn vision mode on or off so the model can see images |
| `/search` | Turn the conversation search bar on or off, like `Ctrl+F` |
| `/clear` | Clear the conversation history |
| `/exit` | Exit localcode |

`/search` is a valid command, but it does not appear in the `/` palette. `Ctrl+F` is the main way to open it. Typing `/search` opens or closes the same search bar.

You can also use `/quit` instead of `/exit`, and `/copy` to copy the last reply.

`/thinking` does nothing for models without a hidden-reasoning channel. localcode tells you this instead of acting as if the setting worked.
