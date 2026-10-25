---
title: Slash Commands
description: Commands and keys available inside the localcode interface.
---

Type `/` in the prompt to open the command palette. Text that starts with `/` is only a command if the first word is a known command. A path like `/opt/project` is sent to the model as a normal message.

| Command | What it does |
| --- | --- |
| `/btw` | Ask a side question without adding it to the conversation |
| `/cd` | Change working directory |
| `/clear` | Clear session |
| `/compact` | Compact session |
| `/copy` | Copy session transcript |
| `/debug` | View debug info |
| `/decide` | Open typed System One decisions |
| `/diff` | Open diff viewer |
| `/exit` | Exit the app (also `/quit`, `/q`) |
| `/export` | Export session transcript |
| `/help` | Help |
| `/lsp` | Language servers: install or start one |
| `/mcps` | Toggle MCPs |
| `/mode` | Choose Chat or Decisions |
| `/models` | Switch model |
| `/new` | New session |
| `/open` | Open a session or project (also `/projects`, `/project`) |
| `/permissions` | Permission preferences |
| `/plugins` | List plugins |
| `/project-context` | Edit project context |
| `/redo` | Redo an undone message |
| `/reload` | Reload configuration |
| `/rename` | Rename session |
| `/restart` | Restart the service |
| `/server` | Model server connection and port: copy the API key, change the port, stop or start the server |
| `/sessions` | Switch session (also `/resume`, `/continue`) |
| `/settings` | Settings |
| `/skills` | Skills |
| `/speak` | Read aloud / stop reading (also `/say`) |
| `/stats` | Usage statistics |
| `/status` | View status: the loaded model and whether a language server is running |
| `/terminal` | New terminal |
| `/themes` | Switch theme |
| `/thinking` | Expand thinking display (also `/toggle-thinking`) |
| `/undo` | Undo previous message |
| `/variants` | Switch model variant (also `/effort`) |
| `/vision` | Vision: download this model's image projector |
| `/worktrees` | Manage workspaces |

## Shell commands

Start a line with `!` to run a shell command yourself, for example `!git status`. The output appears in the session and the model is not involved.

## Keys

The leader key is `ctrl+x`. Press it, then a second key. For example `ctrl+x` then `v` starts voice input. `/help` lists every binding.

Hold **Space** to dictate. The words appear in the prompt as you say them, after whatever was already there, and the last few can change as more is heard. Release to finish. The first use asks for consent and lists what it needs to download: a recorder runtime of about 60 MB and a speech model of about 514 MB. It uses the Mac's default microphone.

Dictation runs on this Mac and shares the GPU with the coding model, so it gives way to it: while the model is reading or writing an answer nothing is transcribed, and the text arrives when the model finishes or when you release. Otherwise it re-transcribes only the phrase you are still speaking, about once a second.

## Notes

- `/lsp` lists language servers. Installing one downloads it only after you confirm.
- `/vision` shows the projector size first and downloads only after you confirm. It applies to the loaded model only.
- `/thinking` only changes how thinking is shown. Hidden reasoning is off by default for every model.
- `/models` opens the picker described in [Models](/localcode/start-here/choose-a-model).
