# Permissions

LocalCode is a coding agent that can read, write, and execute on your machine. Permissions control what it's allowed to do without asking.

## Defaults

| Action | Default |
|--------|---------|
| Read files | Allowed |
| Write / edit files | Allowed (shows diff) |
| Run shell commands | Ask first |
| Delete files | Ask first |
| Install packages | Ask first |
| Git push | Ask first |

The model can always read and write files in your project. Destructive or external actions require confirmation.

## Autonomy levels

Set with `/autonomy`:

| Level | Behavior |
|-------|----------|
| `suggest` | Shows proposed changes, waits for approval |
| `auto_edit` | Applies file edits automatically, asks before shell commands |
| `full_auto` | Does everything without asking |

!!! warning
    `full_auto` runs shell commands without confirmation. Only use this if you trust the task and have version control.

## Project boundaries

LocalCode works within the current directory by default. It won't read or modify files outside your project unless you explicitly ask.

## What you should know

- **Review diffs** — even in auto mode, check what changed with `/changes` or `/diff`
- **Keep secrets out of context** — don't `/add` files with API keys or credentials
- **Use `/undo`** — if something goes wrong, revert instantly

!!! tip
    Start with the default autonomy level. Escalate to `auto_edit` once you're comfortable with how the model works.
