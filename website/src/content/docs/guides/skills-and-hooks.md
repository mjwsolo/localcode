---
title: Skills & Hooks
description: Reusable prompt templates, and shell commands that run at lifecycle events.
---

## Skills

A skill is a Markdown file — a reusable prompt template the model can pull in.
Frontmatter is optional; a file with none is treated as the whole body.

localcode looks for skills in several places, so skills you already wrote for
another agent are picked up:

```text
~/.localcode/skills/
<repo>/.localcode/skills/
<repo>/.agents/skills/      ~/.agents/skills/
<repo>/.claude/skills/      ~/.claude/skills/
<repo>/.opencode/skills/    ~/.config/opencode/skills/
```

Inside the TUI:

```text
/skills     # list loaded skills and where each came from
```

Skills can also be installed from a URL, which is a network fetch — see
[Network Boundary](/localcode/concepts/network-boundary).

## Hooks

Hooks are shell commands that run at lifecycle events. They are defined in
`.localcode/hooks.toml` (project) or `~/.localcode/hooks.toml` (global):

```toml
[hooks]
session_start = "echo 'localcode started' >> /tmp/localcode.log"
user_prompt_submit = "echo '$PROMPT' >> /tmp/prompts.log"
pre_tool_use = "if [ \"$TOOL_NAME\" = 'bash' ]; then echo \"bash: $TOOL_ARGS\" >> /tmp/tools.log; fi"
post_tool_use = ""
```

| Hook | When | Can block? |
| --- | --- | --- |
| `session_start` | Once, at startup | no |
| `user_prompt_submit` | When you submit a prompt | yes |
| `pre_tool_use` | Before each tool call | yes — exit non-zero |
| `post_tool_use` | After each tool call | no |

Available environment variables: `$LOCALCODE_SESSION_ID`,
`$LOCALCODE_REPO_ROOT`, `$LOCALCODE_MODEL`; `$PROMPT`; `$TOOL_NAME`,
`$TOOL_ARGS`; and `$TOOL_RESULT`, `$TOOL_ERROR` for `post_tool_use`.

A tool call blocked by a hook surfaces as error code `E2111`.

### Project hooks must be trusted first

A repo's `.localcode/hooks.toml` runs shell commands on your machine, so
localcode does **not** load it just because you opened the directory. Review it
and trust it explicitly:

```text
/hooks      # show this repo's hooks.toml and trust it
```

Treat trusting a hooks file the same way you'd treat running a script from that
repository — because that is what it is.
