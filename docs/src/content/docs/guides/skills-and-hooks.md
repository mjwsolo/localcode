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
/skills     # list the skills that were found and where each came from; space turns one on or off
```

In `/skills` each row shows `[x]` (on) or `[ ]` (off), where the skill came from and roughly how many tokens its description adds to every request. A skill that is off is not offered to the model and cannot be loaded, until you turn it on again. A skill with `disable-model-invocation: true` in its frontmatter is never offered to the model on its own, but you can still use it by name.

### Bring skills from Claude Code

localcode does not read Claude Code's skill folders on its own: they are written for Claude, and a long list of skills costs tokens in every request. To reuse some, copy the ones you want:

```text
localcode mcp import --from claude
```

You choose from a list that shows each skill's size, roughly how many tokens its description adds to every request, and whether the folder contains scripts. Chosen skills are copied (not linked) into `~/.config/localcode-agent/skills/<name>/`; your Claude Code folders are left alone. Review a skill that contains scripts before you rely on it, since the model may run them. Removing a copied skill is deleting its folder.

Skills are read from disk only. localcode does not fetch skills from a URL.

## Hooks

localcode has no user-defined lifecycle hooks. The discipline plugin runs the project's own checks after edits and feeds failures back to the model; see [Architecture](/localcode/concepts/architecture). To run something on every commit, use your repository's own pre-commit hooks; the agent's `git commit` goes through them like yours does.
