---
title: Permissions
description: What the agent can do on its own, what it must ask about, and what is never allowed.
---

Permissions in localcode are three layers stacked on top of each other. The
bottom layer cannot be turned off.

## 1. Autonomy level — what gets auto-approved

| Level | Reads | File writes | Bash | Installs | Git |
| --- | --- | --- | --- | --- | --- |
| `suggest` | auto | ask | ask | ask | ask |
| `auto_edit` *(default)* | auto | auto | ask | ask | ask |
| `full_auto` | auto | auto | auto | auto | auto |

Set it for a session with the environment variable:

```sh
LOCALCODE_AUTONOMY=suggest localcode
```

`localcode run` (headless) forces `full_auto`, because there is no human
present to answer an approval prompt.

## 2. Session approvals — ask once, remember

When a tool needs approval you can approve it for the rest of the session
rather than for a single call. `/permissions` toggles command approvals on and
off from inside the TUI.

## 3. The safety layer — never overridable

Some operations are refused regardless of autonomy level or approval, including
recursive deletes rooted at `/` or `~`, `mkfs`, raw `dd`, piping a download
straight into a shell, force-pushes to `main`/`master`, `git reset --hard
origin`, and SQL `DROP TABLE` / `DROP DATABASE`.

A second list triggers confirmation rather than a block — `rm`, `git push`,
`git reset`, `pip install`, `npm install`, `mv`, `docker`, `kubectl`.

Writes to sensitive paths are also gated: `.env` files, SSH private keys,
`~/.aws/credentials`, `.npmrc`, `.git/config` and similar.

## Hooks can veto too

A repo's `.localcode/hooks.toml` can define a `pre_tool_use` hook that blocks a
tool call by exiting non-zero. Because hooks run shell commands, a project's
hook file is not loaded until you explicitly trust it with `/hooks`. See
[Skills & Hooks](/localcode/guides/skills-and-hooks).

## Related

- [First Change](/localcode/start-here/first-change)
- [Network Boundary](/localcode/concepts/network-boundary)
