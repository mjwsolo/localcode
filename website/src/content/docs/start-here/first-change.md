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
`pytest -q` (or `npm test`, or `cargo test -q`) comes back red, localcode reads
the failure and edits again rather than declaring success.

The turn ends on evidence — a passing check — not on the model sounding
finished. See [Verification](/localcode/concepts/verification).

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
