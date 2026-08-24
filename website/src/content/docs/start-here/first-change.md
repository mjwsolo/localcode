---
title: Getting Started
description: Install localcode, open a repo, pick a model, make one change.
---

Five steps, each one a real recording from the app.

## 1. Install

```sh
pip install -U localcode
```

Requires macOS on Apple Silicon and Python 3.10+.

## 2. Open your repo

```sh
cd ~/work/some-project
localcode
```

Choose a project whose tests already pass. localcode uses your repo's own checks as proof.

## 3. Choose a model

The model picker opens on your first launch. Use the arrow keys to choose a model. Then press Enter. The star shows the recommended model for your Mac.

![The localcode model picker: seven models, moving down the list and choosing one](/localcode/demo/step-2-choose-model.gif?v=a0c3cc9d)

Learn more in [Choose a Model](/localcode/start-here/choose-a-model).

## 4. Start building

Enter your request in the chat screen. Include the file name and the check to run.

![Entering a goal in the localcode chat screen and pressing Enter](/localcode/demo/step-3-ask.gif?v=0523fe0c)

```text
> Implement the retry decorator in retry.py so every test in test_retry.py
  passes. Do not modify test_retry.py. Then run: pytest -q
```

## 5. Watch it verify

The model reads the stub and tests. It writes the code, runs `pytest -q`, and reports what it checked.

![localcode reading files, editing them, and then showing 5 passed in pytest](/localcode/demo/step-4-verify.gif?v=92c546ff)

<small>Qwen3.6-35B-A3B (IQ2_M) runs locally on `127.0.0.1:8081`. It uses four tool
calls, takes 11.5&nbsp;s, and uses 276 tokens. The repository's tests fail before the turn and
pass after it. The frames come directly from the running app.</small>

Then check the diff yourself:

```sh
git diff
```

`/undo` reverses the last change. `/undo all` reverses the whole session.

## Commands you'll use

| Command | What it does |
| --- | --- |
| `/status` | Shows server health, the current model, and performance settings |
| `/model` | Lists models or switches models (`/model qwen`) |
| `/permissions` | Turns command approvals on or off |
| `/undo` | Reverses the last file change (`/undo all` reverses every change) |
| `/clear` | Clears the conversation history |
| `/exit` | Quits |

See the full list: [Slash Commands](/localcode/reference/slash-commands).

## Two things to know

**Approvals.** By default, localcode uses **auto_edit**. It allows file edits. It asks for confirmation before running commands on a fixed list. This list includes `pip install`, `npm install`, `python `, `git push`, `rm -r`, and `sudo `. It uses a simple substring match. So `python -m pytest` needs approval, but plain `pytest` does not. See [Permissions](/localcode/start-here/permissions).

**The evidence gate.** If an edit turn changes code, localcode must see a successful build, typecheck, test, or lint check for the current files. Otherwise, it will not report success. It ends the turn by saying the task is incomplete. A fixed list of verbs marks a turn as an edit: `fix`, `change`, `edit`, `update`, `refactor`, `rename`, `remove`, and `add`. Requests without these verbs run as general tasks without this gate.

## Next

- [Choose a Model](/localcode/start-here/choose-a-model) - find the best model that fits on your Mac.
- [Network Boundary](/localcode/concepts/network-boundary) - learn what leaves your machine and when.
