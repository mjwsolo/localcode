---
title: Getting Started
description: Install localcode, open a repo, pick a model, make one change.
---

Five minutes from install to a passing test.

## 1. Install

```sh
pip install -U localcode
```

Requires macOS on Apple Silicon and Python 3.10 or newer.

## 2. Open your repo

```sh
cd ~/work/some-project
localcode
```

Start with a project whose tests already pass. localcode uses your tests to check its own work.

## 3. Choose a model

The model picker opens on first launch. The star marks the model recommended for your Mac. Use the arrow keys and press Enter.

![The localcode model picker](/localcode/demo/step-2-choose-model.gif?v=a0c3cc9d)

More in [Choose a Model](/localcode/start-here/choose-a-model).

## 4. Ask for a change

Say which file to change and which check to run.

![Typing a goal in the chat screen](/localcode/demo/step-3-ask.gif?v=0523fe0c)

```text
> Implement the retry decorator in retry.py so every test in test_retry.py
  passes. Do not modify test_retry.py. Then run: pytest -q
```

## 5. Watch it work

localcode reads the files, writes the code, runs `pytest -q` and tells you what it checked.

![localcode reading, editing, then pytest reporting 5 passed](/localcode/demo/step-4-verify.gif?v=92c546ff)

Then review the diff yourself:

```sh
git diff
```


## Commands you will use

| Command | What it does |
| --- | --- |
| `/status` | Server health, current model, performance settings |
| `/model` | List or switch models (`/model qwen`) |
| `/permissions` | Turn command approvals on or off |
| `/clear` | Clear the conversation |
| `/exit` | Quit |

Full list: [Slash Commands](/localcode/reference/slash-commands).

## Approvals

By default localcode edits files without asking and asks before risky commands: installs, `git push`, `rm -r`, `sudo`, and running scripts with `python` or `node`. You can approve once or for the rest of the session. See [Permissions](/localcode/start-here/permissions).

## Next

- [Choose a Model](/localcode/start-here/choose-a-model)
- [Network Boundary](/localcode/concepts/network-boundary)
