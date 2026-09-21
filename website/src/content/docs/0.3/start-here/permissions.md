---
title: Permissions
description: What the agent can do on its own, what it asks about, and what is
  never allowed.
slug: 0.3/start-here/permissions
---

localcode has three autonomy levels. Set one at startup or toggle approvals with `/permissions`:

* **suggest** - asks before every shell command and every file write.
* **auto\_edit** (the interactive default) - edits files without asking, but confirms risky or destructive shell commands such as `rm -rf`, `git push`, `pip install`, `npm install`, and `curl ... | sh`.
* **full\_auto** - nothing prompts.

Two rules hold at every level:

* **Network tools never prompt.** `web_search`, `web_fetch`, and MCP tools run without asking, even in `suggest`. See [Network Boundary](/localcode/0.3/concepts/network-boundary).
* **A hard safety block cannot be turned off.** Catastrophic operations - `rm -rf /`, `mkfs`, `dd` to a disk device, or writing credential files like `~/.ssh/id_rsa` - are refused in every mode, including `full_auto`. It is a guard against mistakes, not a security boundary.
