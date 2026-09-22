---
title: Permissions
description: What the agent can do on its own, what it asks about, and what is never allowed.
---

File edits and shell commands go through the runtime's permission prompt. When the agent wants to edit a file or run a command, the prompt shows what it is about to do and offers three answers:

- **allow once** - run this one action.
- **always** - allow this kind of action for the rest of this session without asking.
- **deny** - refuse it. The agent gets the refusal as a tool result and plans around it.

Reading files does not prompt. `/permissions` shows the current rules for the session.

## Rules that always hold

- **Writes outside the project directory are refused.** The agent can only edit files under the directory you opened. This holds in the interface and in headless runs.
- **Headless runs cannot answer prompts.** `localcode run` has nobody to ask, so it runs the previous 0.3 agent loop in full-auto; out-of-workspace writes are auto-rejected. See [CLI](/localcode/reference/cli).
- **Network tools do not prompt.** `websearch`, `webfetch`, and MCP tools run when the model calls them. See [Network Boundary](/localcode/concepts/network-boundary).

## Project rules

A project `localcode.json` can pre-answer permissions, for example to allow `pytest` without asking or deny a command outright. See [Configuration](/localcode/reference/configuration).

The permission prompt is a guard against mistakes. It is not a security boundary against a hostile model or repository.
