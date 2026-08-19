---
title: Slash Commands
description: Commands available inside the localcode TUI.
---

Type `/` in the chat input to open the command palette. Input starting with `/`
is only treated as a command when the first token is a recognised command name
— a path like `/Users/you/project` is sent to the model as an ordinary message.

`!` at the start of a line enters shell mode.

| Command | What it does |
| --- | --- |
| `/permissions` | Toggle command approvals on/off |
| `/status` | Show runtime: server health, current model, perf config |
| `/restart` | Restart the model server (use when `/status` shows "unreachable") |
| `/mcp` | List or reload MCP servers from `~/.localcode/mcp.json` |
| `/skills` | List loaded skills and where they came from |
| `/model` | List available models / switch (e.g. `/model qwen`) |
| `/delete` | Delete a downloaded model to free disk space (asks first) |
| `/hooks` | Show this repo's `.localcode/hooks.toml` and trust it (runs shell) |
| `/paste` | Attach an image or screenshot from the clipboard (or press `Ctrl+G`) |
| `/thinking` | Show or set the hidden-reasoning policy (`off` / `auto`) |
| `/sounds` | Toggle completion and approval notification sounds |
| `/voice` | Toggle voice mode (push-to-talk dictation into the input box) |
| `/audio` | Toggle audio output (replies read aloud via macOS `say`) |
| `/vision` | Toggle vision mode (let the model see images) |
| `/search` | Toggle the in-conversation search bar (same as `Ctrl+F`) |
| `/undo` | Revert the last file change the agent made (`/undo all` for every change) |
| `/clear` | Clear conversation history |
| `/exit` | Exit localcode |

`/search` is recognised but is not listed in the `/` palette, because `Ctrl+F`
is the canonical way in; typing it toggles the same search bar.

Also accepted: `/quit` (same as `/exit`), `/image` (same as `/paste`), and
`/copy`.

`/thinking` has no effect on models with no hidden-reasoning channel; localcode
says so rather than pretending the setting took.
