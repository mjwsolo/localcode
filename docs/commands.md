# Commands

Slash commands give you direct control. The model handles most things automatically, but commands let you be explicit.

## Mode

| Command | What it does |
|---------|-------------|
| `/switch` | Toggle between fast and reasoning mode |

```
/switch  →  Switched to reasoning mode (26 tok/s, thinking on)
/switch  →  Switched to fast mode (27 tok/s, thinking off)
```

## Navigation

| Command | What it does |
|---------|-------------|
| `/help` | Show all commands |
| `/status` | Runtime info — model, mode, repo |
| `/model` | List or switch models |
| `/files` | List repo files in context |
| `/read <file>` | Read a file into context |
| `/add <file>` | Pin a file to context (persists across turns) |
| `/drop <file>` | Unpin a file |
| `/context` | Show current context usage |
| `/diff` | Show git diff |

## Editing

| Command | What it does |
|---------|-------------|
| `/apply` | Apply a pending code patch |
| `/undo` | Revert last file change |
| `/undo all` | Revert all changes this session |
| `/changes` | List files changed this session |
| `/shell <cmd>` | Run a shell command |
| `/bg <cmd>` | Run a shell command in background |
| `/verify` | Syntax check / run tests on last file |
| `/verify <cmd>` | Run a custom verification command |

## Session

| Command | What it does |
|---------|-------------|
| `/thinking` | Show or set thinking mode visibility (`hidden`, `summary`, `full`) |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

!!! tip
    Most of the time you don't need commands. Just describe what you want and the model will use the right tools. Commands are for when you want precise control.
