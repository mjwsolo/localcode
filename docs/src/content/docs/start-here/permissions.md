---
title: Permissions
description: What the agent does on its own, what it asks about, and where the boundary is.
---

Inside the project you opened, the agent works on its own. It reads files, edits files and runs shell commands without asking first. That is the default, and it is what lets a task run to the end unattended.

A few actions stop and ask. The prompt shows what the agent is about to do and offers three answers:

- **allow once** - run this one action.
- **always** - allow this kind of action for the rest of this session without asking.
- **deny** - refuse it. The agent gets the refusal as a tool result and plans around it.

## What asks

| Action | Why |
| --- | --- |
| Reading or writing outside the project directory | The project is the boundary |
| Reading a `.env` file | It usually holds secrets |
| Repeating the same tool call over and over | It is probably stuck |

`/permissions` has one switch, **Auto-approve permissions**. It is off by default. Turning it on answers "allow once" to these prompts for you.

## What does not ask

- **Edits and shell commands inside the project.** They run when the model calls them.
- **Network tools.** `websearch`, `webfetch` and MCP tools run when the model calls them. See [Network Boundary](/localcode/concepts/network-boundary).

So open a project you are willing to let the agent change, and keep it under version control. Undo is `git`.

## Tightening it

A project `localcode.json` can make edits or commands ask, or deny a command outright:

```json
{
  "permission": {
    "bash": "ask",
    "edit": "ask"
  }
}
```

It can also allow or deny by pattern, for example allowing `pytest` while asking for everything else. See [Configuration](/localcode/reference/configuration).

## What this is not

The prompts guard against mistakes. They are not a security boundary against a hostile repository or a model that has been steered by text it read. A shell command can reach anything your user account can. Do not open a repository you do not trust.
