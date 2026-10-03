---
title: Slash Commands
description: Commands and keys available inside the localcode interface.
---

Type `/` in the prompt to open the command palette. Text that starts with `/` is only a command if the first word is a known command. A path like `/opt/project` is sent to the model as a normal message.

| Command | What it does |
| --- | --- |
| `/compact` | Compact session |
| `/copy` | Copy session transcript |
| `/debug` | View debug info |
| `/diff` | Open diff viewer |
| `/editor` | Open the external editor for the prompt (falls back to `nano`) |
| `/exit` | Exit the app |
| `/export` | Export session transcript |
| `/help` | Help |
| `/lsp` | Language servers: install or start one |
| `/mcps` | Toggle MCPs |
| `/models` | Switch model |
| `/move` | Move to another project dir |
| `/new` | New session |
| `/permissions` | Permissions |
| `/project-context` | Edit project context |
| `/rename` | Rename session |
| `/review` | Review changes `[commit\|branch\|pr]`, defaults to uncommitted |
| `/sessions` | Switch session |
| `/settings` | Settings |
| `/skills` | Skills |
| `/speak` | Read aloud / stop reading |
| `/status` | View status |
| `/themes` | Switch theme |
| `/thinking` | Expand thinking display |
| `/timestamps` | Hide timestamps |
| `/undo` | Undo previous message |
| `/vision` | Vision: download this model's image projector |

## Shell commands

Start a line with `!` to run a shell command yourself, for example `!git status`. The output appears in the session and the model is not involved.

## Keys

The leader key is `ctrl+x`. Press it, then a second key. For example `ctrl+x` then `v` starts voice input. `/help` lists every binding.

Hold **Space** on an empty prompt to record voice. Release to transcribe into the prompt. The first use asks for consent and lists what it needs to download: a recorder runtime of about 60 MB and a speech model of about 514 MB. Set `LOCALCODE_MIC` to choose the microphone.

## Notes

- `/lsp` lists language servers. Installing one downloads it only after you confirm.
- `/vision` shows the projector size first and downloads only after you confirm. It applies to the loaded model only.
- `/thinking` only changes how thinking is shown. Hidden reasoning is off by default for every model.
- `/models` opens the picker described in [Models](/localcode/start-here/choose-a-model).
