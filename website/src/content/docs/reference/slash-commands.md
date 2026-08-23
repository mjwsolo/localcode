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
| `/delete` | Delete a downloaded model to free disk space (asks first) |
| `/restart` | Restart the model server |
| `/permissions` | Turn command approvals on or off |
| `/undo` | Undo the last file change. `/undo all` undoes every change this session |
| `/thinking` | Show or set hidden reasoning: `off` or `auto` |
| `/vision` | Let the model see images |
| `/paste` | Attach an image from the clipboard (or press `Ctrl+G`) |
| `/mcp` | List MCP servers, or reload them after editing `mcp.json` |
| `/skills` | List loaded skills and where they came from |
| `/hooks` | Show this project's `hooks.toml` and trust it |
| `/voice` | Push-to-talk dictation into the input box |
| `/audio` | Read replies aloud |
| `/sounds` | Completion and approval sounds |
| `/search` | Search the conversation (or press `Ctrl+F`) |
| `/clear` | Clear the conversation |
| `/exit` | Quit |

`/quit` is the same as `/exit`, and `/image` is the same as `/paste`.

`/thinking` has no effect on models without a hidden-reasoning channel; localcode tells you when that is the case.
