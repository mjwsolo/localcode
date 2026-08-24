---
title: Permissions
description: What the agent can do on its own, what it must ask about, and what is never allowed.
---

Permissions in localcode have three layers. The bottom layer cannot be turned off.

## 1. What the gate checks

The gate only checks **tools that run shell commands** (`bash`, `background_process`) and **tools that write files** (`write_file`, `append_file`, `edit_file`, `multi_edit`, `edit_diff`).

All other tools run without a prompt at every autonomy level. These include `read_file`, `grep`, and, importantly, **`web_search`, `web_fetch`, and all MCP tools**. `suggest` mode does not block network access. See [Network Boundary](/localcode/concepts/network-boundary).

## 2. Autonomy levels

| Level | File writes | `bash` | `background_process` |
| --- | --- | --- | --- |
| `suggest` | confirmed | every command confirmed | confirmed |
| `auto_edit` *(interactive default)* | no prompt | confirmed if it matches the pattern lists | confirmed |
| `full_auto` | no prompt | no prompt | no prompt |

**`background_process` is confirmed** outside `full_auto`, no matter what the command does. It sends a raw string directly to `/bin/sh`. It does not use the substring shortcuts that `bash` uses. There is one exception. The session-approval check runs before the always-confirm rule. If you already approved the command's leading token in this session, it runs without another prompt.

At `auto_edit`, a `bash` command is confirmed if it matches either of these lists:

- **Risky shell patterns** - sending a `curl`/`wget` download into a shell, `git push --force`, `sudo rm`, or `git reset --hard origin`.
- **The destructive substring list** - this list is broad and handles most cases: `rm -rf`, `rm -r`, `rmdir`, `git push`, `git reset --hard`, `sudo `, `pip install`, `npm install`, `brew install`, `docker rm`, `kubectl delete`, `DROP TABLE`, `DELETE FROM`, `python `, `python3 `, `node `, `npm run`, `npm start`.

So `pip install`, `npm install`, and `npm run` **do** prompt at `auto_edit`. Matching uses plain substrings, so it also matches inside longer command lines.

Commands that are not on either list do not prompt at this level. Examples include `ls`, `cat`, `grep`, `pytest`, `cargo test`, and a plain `curl https://…`. Only `curl … | sh` is caught. To stop before every command, use `suggest`.

Writes to the agent's per-session notebook scratch directory never prompt at any level.

## 3. Session approvals - ask once, remember

When you confirm a command, you can approve its leading token for the rest of the session instead of approving only one call. Use `/permissions` inside the TUI to turn command approvals on or off.

## 4. The safety layer - never overridable

A hard block runs before every tool call in **all** modes, including `full_auto` and headless. It blocks operations that have no valid use for an agent.

For shell commands, the patterns are narrow. They are tied to real device, root, and home targets. This means that searching for the text of a dangerous command does not trigger them. **Examples** (not the full list):

- `rm -rf` / `rm -fr` aimed at `/`, `~`, or `$HOME`
- `mkfs`, `wipefs`
- `dd … of=/dev/…`, and redirects into `/dev/sd*`, `/dev/nvme*`, `/dev/disk*`
- `chmod -R 777 /`
- redirects into `/etc/`
- the classic `:(){ :|:& };:` fork bomb

SQL patterns such as `DROP TABLE` are deliberately *not included*. Bash does not run SQL, and blocking these patterns made it impossible to search for the text. They are still on the `auto_edit` confirmation list above. In the same way, `curl … | sh`, force-pushes, and `sudo rm` require confirmation instead of being blocked, so you can approve them.

For file writes, the block protects credential files.

Blocked write targets are matched in two ways. The first is an exact **basename**: `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`, `authorized_keys`, `known_hosts`, `.netrc`, `.npmrc`, `.pypirc`, `credentials`, `credentials.json`, `shadow`, `passwd`, or `sudoers`. The second is an exact **path segment**: `.ssh`, `.aws`, or `.gnupg`.

Segment matching checks one path part at a time. This means only single-part entries can match. For example, anything under `~/.config/gcloud` is **not** covered by this list.

These actions are refused, not confirmed. No autonomy level or session approval can turn the block off. It is a guard against dangerous mistakes, not a security boundary. The matching uses substrings and regular expressions, so a determined caller can work around it.

## Related

- [Getting Started](/localcode/start-here/first-change)
- [Network Boundary](/localcode/concepts/network-boundary)
