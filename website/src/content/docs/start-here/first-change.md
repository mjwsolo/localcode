---
title: First Change
description: Make one small, verified edit in a repo you already have.
---

The fastest way to understand localcode is to give it one small change in a
repo you already know, and watch it prove the change works.

## 1. Open a repo you know

```sh
cd ~/work/some-project
localcode
```

Pick a project with a test command that already passes. localcode uses your
repo's own checks as its evidence, so a repo that is already green makes the
loop easy to read.

On first launch you'll step through setup, a mode choice (**Fast** or
**Reasoning**), and the model picker. After that, `localcode` drops you
straight into the chat screen.

## 2. Ask for one specific thing

Small and concrete beats broad and vague — especially with a local model.

```text
> in src/timeutil.py, make parse_duration accept compound units like "1h30m"
```

Good first asks:

- "add a `--json` flag to the `report` command"
- "this function returns `None` on empty input — raise `ValueError` instead"
- "write a test for the branch in `parse_config` that handles a missing key"

## 3. Watch the loop

A turn is a sequence of tool calls, and localcode shows each one:

- `read_file` / `grep` / `list_files` — building context before touching anything
- `edit_file` / `multi_edit` / `write_file` — the actual change
- `bash` — running your repo's checks

By default localcode runs at the **auto_edit** autonomy level: reads and file
edits are auto-approved, while shell commands and installs stop and ask you
first. So the `bash` step that runs your tests is the point where you'll be
prompted. See [Permissions](/localcode/start-here/permissions).

## 4. Expect a failure, and expect a recovery

A first attempt that fails a check is normal and is the interesting part. When
the test command comes back red, the failure output goes back into the turn and
the model edits again rather than declaring success.

Be clear about who does what here. The model chooses to run your tests;
localcode does not run them for you. What localcode enforces is the *claim*: on
a turn that changed code, if it never observed a build, typecheck, test or lint
command that passed against the current file contents, it will not report
success — it closes the turn saying the task remains incomplete instead. On
build-shaped goals it additionally re-runs the project's own typecheck itself
and refuses to end the turn while that is red.

See [Verification](/localcode/concepts/verification) for exactly where the hard
gate sits.

## 5. Review the diff

localcode is not a replacement for reading your own diff:

```sh
git diff
```

If you don't want the change, `/undo` reverts the last file change the agent
made, and `/undo all` reverts every change from the session. See
[Undo](/localcode/guides/undo).

## 6. Useful commands while you work

| Command | What it does |
| --- | --- |
| `/status` | Server health, current model, perf configuration |
| `/model` | List models or switch (`/model qwen`) |
| `/permissions` | Toggle command approvals on and off |
| `/undo` | Revert the last file change (`/undo all` for every change) |
| `/clear` | Clear conversation history |
| `/exit` | Quit |

The full list is in [Slash Commands](/localcode/reference/slash-commands).

## When you're done

localcode prints a session ID on exit. To pick the conversation back up:

```sh
localcode --resume last
```

## Next

- [Choose a Model](/localcode/start-here/choose-a-model) — get the best model your Mac can hold.
- [Network Boundary](/localcode/concepts/network-boundary) — what leaves the machine, and when.
