---
title: Skills
description: Reusable instructions the model can load into a task.
---

## Skills

A skill is a folder with a `SKILL.md` file. The file holds instructions the model can load when a task calls for them: a release checklist, a migration recipe, the way your team writes tests. The folder can hold supporting files the instructions refer to.

localcode reads skill folders from two places:

```text
<project>/.localcode-agent/skills/      # this project
~/.config/localcode-agent/skills/       # every project
```

In the interface:

```text
/skills     # list the skills that were found and where each came from
```

Skills are read from disk only. localcode does not fetch skills from a URL.

## Hooks

Lifecycle hooks are a feature of the 0.3 classic interface (`~/.localcode/hooks.toml`) and are not part of the 0.4 interface. In 0.4, the discipline plugin runs the project's own checks after edits; see [Architecture](/localcode/concepts/architecture). For the classic hook format, use the 0.3 docs through the version switcher in the header.
