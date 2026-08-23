---
title: Permissions
description: What localcode does on its own, what it asks about, and what it never does.
---

localcode has three autonomy levels and one set of hard limits that apply at every level.

## Autonomy levels

| Level | File edits | Shell commands |
| --- | --- | --- |
| `suggest` | asks every time | asks every time |
| `auto_edit` (default) | no prompt | asks before risky commands |
| `full_auto` | no prompt | no prompt |

Set the level for a session with an environment variable:

```sh
LOCALCODE_AUTONOMY=suggest localcode
```

`/permissions` inside the app toggles command approvals. `localcode run` (headless) always uses `full_auto`, because nobody is there to answer a prompt.

## What counts as risky

At `auto_edit`, localcode asks before a command that:

- installs something: `pip install`, `npm install`, `brew install`
- pushes or rewrites history: `git push`, `git reset --hard`
- deletes things: `rm -r`, `rm -rf`, `rmdir`, `docker rm`, `kubectl delete`
- runs as root: `sudo`
- runs a script or server: `python`, `python3`, `node`, `npm run`, `npm start`
- pipes a download into a shell, such as `curl ... | sh`
- contains `DROP TABLE` or `DELETE FROM`

Anything else runs without a prompt: `ls`, `grep`, `pytest`, `cargo test`, `git status`, a plain `curl`. Starting a long-running process in the background always asks.

When localcode asks, you can approve this one command or every command that starts with the same word for the rest of the session.

## Tools that never ask

Reading files, searching code, `web_search`, `web_fetch` and MCP tools run without a prompt at every level, including `suggest`. If you add an MCP server, you are trusting it. See [Network Boundary](/localcode/concepts/network-boundary).

## What localcode never does

These are refused at every level, including `full_auto`, and cannot be approved:

- Shell commands that would wipe the disk or the system: `rm -rf /` or `rm -rf ~`, `mkfs`, `wipefs`, `dd` onto a device, writing into `/dev/` or `/etc/`, `chmod -R 777 /`, fork bombs.
- Writing to credential files: SSH keys, `authorized_keys`, `known_hosts`, `.netrc`, `.npmrc`, `.pypirc`, `credentials`, `shadow`, `passwd`, `sudoers`, and anything under `.ssh`, `.aws`, `.gnupg` or `.config/gcloud`.

This is a guard against mistakes, not a sandbox. Use `suggest` mode when you want to see every command before it runs.

## Hooks can also block

A repository's `.localcode/hooks.toml` can define a `pre_tool_use` hook. The hook can block a tool call by exiting with a non-zero status. Hooks run shell commands, so a project's hook file is not loaded until you explicitly trust it by launching with `LOCALCODE_TRUST_PROJECT_HOOKS=1`. See [Skills & Hooks](/localcode/guides/skills-and-hooks).

## Related

- [Getting Started](/localcode/start-here/first-change)
- [Network Boundary](/localcode/concepts/network-boundary)
