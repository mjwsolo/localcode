---
title: Permissions
description: What the agent does on its own, what it asks about, and where the boundary is.
---

## Project startup code

LocalCode ignores repository configuration by default. This includes discovered
plugins and project MCP startup commands, which can execute code before tool
permissions apply. User-global configuration and LocalCode's packaged plugins
still load.

For a project whose configuration you have reviewed, explicitly enable it:

```bash
localcode --trust-project
```

Trust applies to that launch, including a second window attached to an existing
model server. Repository settings can also change tool permissions. Review them
before enabling trust; ordinary code files remain available for the agent to read
and edit without this flag.

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

`/permissions` has three switches:

- **Auto-approve permissions** (off by default). Turning it on answers "allow once" to the prompts above for you. It never answers the sandbox prompt below.
- **Sandbox shell commands** (on by default). See [The shell sandbox](#the-shell-sandbox).
- **Allow network in sandbox** (on by default). Turn it off to limit sandboxed commands to this machine.

## The shell sandbox

On macOS, every shell command the agent runs goes through the operating system's sandbox (`sandbox-exec`, the same seatbelt mechanism other coding agents use). The operating system enforces it, so it holds for scripts, `python -c "..."` and `cd` inside a command, not only for the paths written in the command line.

Inside the sandbox a command can:

- read and write the project directory, temporary directories and the usual tool caches (npm, pip and uv, cargo, go, bun);
- read system files, installed tools, your git configuration and your shell startup files;
- use the network (unless you turn that switch off, which leaves localhost only).

It cannot write anywhere else in your home directory, and it cannot read or write `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.docker`, the GitHub CLI login or the keychain folders.

When the sandbox blocks a command, LocalCode lists what was blocked and asks whether to run that command outside the sandbox: **allow once**, **always** for that command, or **deny**. Auto-approve never answers this prompt, so leaving the sandbox always takes a deliberate yes from you. Turning **Sandbox shell commands** off in `/permissions` removes the boundary for every command. That is a separate, deliberate switch.

Limits to know about:

- macOS only. On Linux and Windows commands are not sandboxed.
- A blocked command is detected from its failure and the system log. A program that hides its error, or a denial the log does not report, can look like an ordinary failure.
- The sandbox covers shell commands. File tools, web tools and MCP servers keep their own prompts and rules, and MCP servers run outside it.
- Reading files outside the project with a command is limited to the locations above, but anything you allow outside the sandbox runs with your full account.

## What does not ask

- **Edits and shell commands inside the project.** They run when the model calls them. Shell commands run in the sandbox above.
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

The file and path prompts guard against mistakes. The shell sandbox is a real boundary for shell commands on macOS, but it is not a promise that untrusted code is safe: it does not cover MCP servers, project startup code (see above), or anything you allow outside it, and it shares your network when network is on. Do not open a repository you do not trust.
