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

| Level | File writes | `bash` | `background_process` |
| --- | --- | --- | --- |
| `suggest` | confirmed | every command confirmed | confirmed |
| `auto_edit` *(interactive default)* | no prompt | confirmed if it matches the pattern lists | confirmed |
| `full_auto` | no prompt | no prompt | no prompt |

**`background_process` is always confirmed** outside `full_auto`, regardless of
what the command is — it hands a raw string straight to `/bin/sh` with none of
the substring shortcuts `bash` gets. The only escape is `full_auto` or having
already session-approved it.

At `auto_edit`, a `bash` command is confirmed when it matches either of two
lists:

- **Risky shell patterns** — piping a `curl`/`wget` download into a shell,
  `git push --force`, `sudo rm`, `git reset --hard origin`.
- **The destructive substring list**, which is broad and does most of the work:
  `rm -rf`, `rm -r`, `rmdir`, `git push`, `git reset --hard`, `sudo `,
  `pip install`, `npm install`, `brew install`, `docker rm`, `kubectl delete`,
  `DROP TABLE`, `DELETE FROM`, `python `, `python3 `, `node `, `npm run`,
  `npm start`.

So `pip install`, `npm install` and `npm run` **do** prompt at `auto_edit`.
Matching is plain substring, so it fires inside a longer command line too.

What does *not* prompt at this level is anything on neither list — `ls`,
`cat`, `grep`, `pytest`, `cargo test`, or a bare `curl https://…` (only
`curl … | sh` is caught). If you want every command to stop, use `suggest`.

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
to credential material.

Blocked write targets are matched two ways — an exact **basename**
(`id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`, `authorized_keys`,
`known_hosts`, `.netrc`, `.npmrc`, `.pypirc`, `credentials`,
`credentials.json`, `shadow`, `passwd`, `sudoers`) or an exact **path
segment**: `.ssh`, `.aws`, `.gnupg`.

Note that segment matching compares one path component at a time, so only
single-component entries can ever match. Anything under `~/.config/gcloud`, for
instance, is **not** covered by this list.

These are refusals, not prompts. No autonomy level and no session approval
turns them off. It is a footgun guard, not a security boundary.

## Hooks can veto too

A repo's `.localcode/hooks.toml` can define a `pre_tool_use` hook that blocks a
tool call by exiting non-zero. Because hooks run shell commands, a project's
hook file is not loaded until you explicitly trust it with `/hooks`. See
[Skills & Hooks](/localcode/guides/skills-and-hooks).

## Related

- [First Change](/localcode/start-here/first-change)
- [Network Boundary](/localcode/concepts/network-boundary)
