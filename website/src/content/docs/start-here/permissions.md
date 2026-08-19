---
title: Permissions
description: What the agent can do on its own, what it must ask about, and what is never allowed.
---

Permissions in localcode are three layers stacked on top of each other. The
bottom layer cannot be turned off.

## 1. What the gate actually checks

Before autonomy levels, the important fact: the gate only considers
**shell-executing tools** (`bash`, `background_process`) and **file-write
tools** (`write_file`, `append_file`, `edit_file`, `multi_edit`, `edit_diff`).

Everything else is dispatched without a prompt at every autonomy level —
including `read_file`, `grep`, and, notably, **`web_search`, `web_fetch` and
all MCP tools**. `suggest` mode is not a no-network mode. See
[Network Boundary](/localcode/concepts/network-boundary#approvals-the-network-tools-never-ask).

## 2. Autonomy levels

| Level | File writes | Shell commands |
| --- | --- | --- |
| `suggest` | confirmed | every command confirmed |
| `auto_edit` *(interactive default)* | no prompt | only risky-looking commands confirmed |
| `full_auto` | no prompt | no prompt |

At `auto_edit`, "risky-looking" means a match against a small pattern list —
piping a `curl`/`wget` download into a shell, `git push --force`, `sudo rm`,
`git reset --hard origin` — or the destructive-pattern list. An ordinary
`curl`, `pip install` or `npm install` runs **without** asking. If you want
every command to stop, use `suggest`.

Writes into the agent's per-session notebook scratch directory are never
prompted, at any level.

Set the level for a session with the environment variable:

```sh
LOCALCODE_AUTONOMY=suggest localcode
```

`localcode run` (headless) forces `full_auto`, because there is no human
present to answer an approval prompt.

## 3. Session approvals — ask once, remember

When a command is confirmed you can approve that command's leading token for
the rest of the session rather than for a single call. `/permissions` toggles
command approvals on and off from inside the TUI.

## 4. The safety layer — never overridable

A hard block runs before every tool dispatch in **all** modes, including
`full_auto` and headless. It covers operations with no legitimate agent use:
catastrophic shell (`rm -rf /`, `mkfs`, `dd of=/dev/…`, fork bombs) and writes
to credential material — SSH private keys, `authorized_keys`, `.netrc`,
`.npmrc`, `.pypirc`, `credentials.json`, `shadow`/`passwd`/`sudoers`, and
anything under `.ssh`, `.aws`, `.gnupg` or `.config/gcloud`.

These are refusals, not prompts. No autonomy level and no session approval
turns them off.

## Hooks can veto too

A repo's `.localcode/hooks.toml` can define a `pre_tool_use` hook that blocks a
tool call by exiting non-zero. Because hooks run shell commands, a project's
hook file is not loaded until you explicitly trust it with `/hooks`. See
[Skills & Hooks](/localcode/guides/skills-and-hooks).

## Related

- [First Change](/localcode/start-here/first-change)
- [Network Boundary](/localcode/concepts/network-boundary)
