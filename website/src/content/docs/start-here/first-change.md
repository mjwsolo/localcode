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
> in src/timeutil.py, change parse_duration to accept compound units like "1h30m"
```

Start with a verb localcode's goal classifier recognises as an edit — `fix`,
`change`, `edit`, `update`, `refactor`, `rename`, `remove`, `add`. That is what
puts the turn inside the evidence gate described below; *"make parse_duration
accept 1h30m"* classifies as a general task and is not gated.

Good first asks:

- "add a `--json` flag to the `report` command"
- "change `parse_config` to raise `ValueError` on empty input instead of returning `None`"
- "add a test for the missing-key branch in `parse_config`"

## 3. Watch the loop

A turn is a sequence of tool calls, and localcode shows each one:

- `read_file` / `grep` / `list_files` — building context before touching anything
- `edit_file` / `multi_edit` / `write_file` — the actual change
- `bash` — running your repo's checks

By default localcode runs at the **auto_edit** autonomy level: file edits go
through without asking, and so do ordinary shell commands — only
risky-looking ones (piping a download into a shell, a force-push, `sudo rm`)
stop for confirmation. If you want every command to ask, start with
`LOCALCODE_AUTONOMY=suggest localcode`. Note that `web_search`, `web_fetch` and
MCP tools never prompt at any level. See
[Permissions](/localcode/start-here/permissions).

## 4. Expect a failure, and expect a recovery

A first attempt that fails a check is normal and is the interesting part. When
the test command comes back red, the failure output goes back into the turn and
the model edits again rather than declaring success.

Be clear about who does what here. The model chooses to run your tests;
localcode does not run them for you. What localcode enforces is the *claim*,
and only for requests it classified as a build or an edit: if such a turn
changed code files and localcode never observed a build, typecheck, test or
lint command that passed against the current file contents, it will not report
success — it closes the turn saying the task remains incomplete instead. On
build-shaped goals it additionally runs a typecheck itself and refuses to end
the turn while that is red.

A request that classified as a general task carries no such guarantee. See
[Verification](/localcode/concepts/verification) for the exact scope.

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
