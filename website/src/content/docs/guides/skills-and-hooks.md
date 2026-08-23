---
title: Skills & Hooks
description: Reusable prompt files, and shell commands that run at lifecycle events.
---

## Skills

A skill is a Markdown file the model can load as instructions. Frontmatter is optional.

localcode looks for skills in these places, so skills written for other agents work too:

```text
~/.localcode/skills/
<repo>/.localcode/skills/
<repo>/.agents/skills/      ~/.agents/skills/
<repo>/.claude/skills/      ~/.claude/skills/
<repo>/.opencode/skills/    ~/.config/opencode/skills/
```

```text
/skills     # list loaded skills and where each came from
```

You can also install a skill from a URL. That is a network download.

## Hooks

Hooks are shell commands that run at points in a session. Define them in `.localcode/hooks.toml` in a project, or `~/.localcode/hooks.toml` for every project:

```toml
[hooks]
session_start = "echo 'localcode started' >> /tmp/localcode.log"
user_prompt_submit = "echo '$PROMPT' >> /tmp/prompts.log"
pre_tool_use = "if [ \"$TOOL_NAME\" = 'bash' ]; then echo \"bash: $TOOL_ARGS\" >> /tmp/tools.log; fi"
post_tool_use = ""
```

| Hook | When | Can block? |
| --- | --- | --- |
| `session_start` | Once at startup | no |
| `user_prompt_submit` | When you submit a prompt | yes, exit non-zero |
| `pre_tool_use` | Before each tool call | yes, exit non-zero |
| `post_tool_use` | After each tool call | no |

Hooks see `$LOCALCODE_SESSION_ID`, `$LOCALCODE_REPO_ROOT`, `$LOCALCODE_MODEL`, `$PROMPT`, `$TOOL_NAME` and `$TOOL_ARGS`. `post_tool_use` also gets `$TOOL_RESULT` and `$TOOL_ERROR`. A hook that takes longer than 10 seconds is stopped.

A blocked tool call shows error `E2111`.

### Trust project hooks first

A project's `hooks.toml` runs shell commands on your machine, so localcode does not load it just because you opened the directory. Review it, then trust it:

```text
/hooks      # show this repo's hooks.toml and trust it
```

Trusting a hooks file is the same as running a script from that repository.
