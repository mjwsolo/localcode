---
title: Undo
description: Reverting file changes the agent made.
---

localcode snapshots a file before it changes it, so an edit can be taken back
without reaching for git.

```text
/undo         # revert the last file change the agent made
/undo all     # revert every change the agent made this session
```

## What is tracked

The change log records, per change: the path (relative to the repo root),
whether the file existed beforehand, its original contents, a timestamp, and
which tool made the change. Reverting a file the agent *created* removes it;
reverting a file it *modified* restores the previous contents.

The log covers file changes made by the agent's edit and write tools. It does
not cover side effects of shell commands you approved — a `bash` step that
moved a directory or ran a formatter is outside its reach. Git remains the
backstop for those.

## Undo is not your review

`/undo` is for "that wasn't what I meant". Reading the diff is still the job:

```sh
git diff
```

Working on a branch makes both cheaper.

## Related

- [First Change](/localcode/start-here/first-change)
- [Permissions](/localcode/start-here/permissions)
