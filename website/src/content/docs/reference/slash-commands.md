---
title: Slash Commands
description: Commands available inside the localcode TUI.
---

Type `/` in the chat box to open the command palette. Text that starts with `/` is only a command if the first word is a known command. A path like `/Users/you/project` is sent to the model as a normal message.

Start a line with `!` to enter shell mode.

| Command | What it does |
| --- | --- |
| `/permissions` | Turn command approvals on or off |
| `/status` | Show the server health, current model, and performance settings |
| `/restart` | Restart the model server when `/status` shows "unreachable" |
| `/mcp` | List or reload MCP servers from `~/.localcode/mcp.json` |
| `/skills` | List loaded skills and their sources |
| `/model` | List or switch models, for example `/model qwen` |
| `/delete` | Delete a downloaded model to free disk space after asking first |
| `/hooks` | Show this repo's `.localcode/hooks.toml` and trust it so it can run shell commands |
| `/paste` | Attach an image or screenshot from the clipboard, or press `Ctrl+G` |
| `/thinking` | Show or set the hidden-reasoning policy to `off` or `auto` |
| `/sounds` | Turn completion and approval sounds on or off |
| `/voice` | Turn voice mode on or off for push-to-talk dictation in the input box |
| `/audio` | Turn audio output on or off so macOS `say` can read replies aloud |
| `/vision` | Turn vision mode on or off so the model can see images |
| `/search` | Turn the conversation search bar on or off, like `Ctrl+F` |
| `/undo` | Undo the agent's last file change, or use `/undo all` to undo every change |
| `/clear` | Clear the conversation history |
| `/exit` | Exit localcode |

`/search` is a valid command, but it does not appear in the `/` palette. `Ctrl+F` is the main way to open it. Typing `/search` opens or closes the same search bar.

You can also use `/quit` instead of `/exit`, `/image` instead of `/paste`, and `/copy`.

`/thinking` does nothing for models without a hidden-reasoning channel. localcode tells you this instead of acting as if the setting worked.
