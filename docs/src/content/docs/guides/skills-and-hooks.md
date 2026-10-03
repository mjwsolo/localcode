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

localcode has no user-defined lifecycle hooks. The discipline plugin runs the project's own checks after edits and feeds failures back to the model; see [Architecture](/localcode/concepts/architecture). To run something on every commit, use your repository's own pre-commit hooks; the agent's `git commit` goes through them like yours does.
