---
title: Skills & Hooks
description: Reusable prompt templates, and shell commands that run at lifecycle events.
---

## Skills

A skill is a Markdown file. It is a reusable prompt template that the model can use. Frontmatter is optional. If a file has no frontmatter, its full body is used.

localcode finds skills in several places. This means it can use skills you wrote for another agent:

```text
~/.localcode/skills/
<repo>/.localcode/skills/
<repo>/.agents/skills/      ~/.agents/skills/
<repo>/.claude/skills/      ~/.claude/skills/
<repo>/.opencode/skills/    ~/.config/opencode/skills/
```

In the TUI:

```text
/skills     # list loaded skills and where each came from
```

You can also install skills from a URL. This fetches data from the network. See [Network Boundary](/localcode/concepts/network-boundary).

## Hooks

Hooks are shell commands that run during lifecycle events. Define them in `.localcode/hooks.toml` for a project or `~/.localcode/hooks.toml` globally:

```toml
[hooks]
session_start = "echo 'localcode started' >> /tmp/localcode.log"
user_prompt_submit = "echo '$PROMPT' >> /tmp/prompts.log"
pre_tool_use = "if [ \"$TOOL_NAME\" = 'bash' ]; then echo \"bash: $TOOL_ARGS\" >> /tmp/tools.log; fi"
post_tool_use = ""
```

| Hook | When it runs | Can it block? |
| --- | --- | --- |
| `session_start` | Once at startup | no |
| `user_prompt_submit` | When you submit a prompt | yes |
| `pre_tool_use` | Before each tool call | yes — exit non-zero |
| `post_tool_use` | After each tool call | no |

These environment variables are available: `$LOCALCODE_SESSION_ID`, `$LOCALCODE_REPO_ROOT`, `$LOCALCODE_MODEL`, `$PROMPT`, `$TOOL_NAME`, `$TOOL_ARGS`, and `$TOOL_RESULT`, `$TOOL_ERROR` for `post_tool_use`.

If a hook blocks a tool call, you will see error code `E2111`.

### Trust Project Hooks First

A repo's `.localcode/hooks.toml` runs shell commands on your machine. Because of this, localcode does **not** load it when you only open the directory. Review it, then opt in explicitly:

```bash
LOCALCODE_TRUST_PROJECT_HOOKS=1 localcode
```

Trusting a hooks file is the same as running a script from that repository. Treat it with the same care.
