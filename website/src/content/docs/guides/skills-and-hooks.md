---
title: Skills
description: Reusable prompt templates the model can load into a task.
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
