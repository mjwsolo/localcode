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
pass after it.</small>

## Key commands

| Command | What it does |
| --- | --- |
| `/status` | Shows server health, the current model, and performance settings |
| `/model` | Lists models or switches models (`/model qwen`) |
| `/permissions` | Turns command approvals on or off |
| `/clear` | Clears the conversation history |
| `/exit` | Quits |

See the full list: [Slash Commands](/localcode/reference/slash-commands).


## Next

- [Choose a Model](/localcode/start-here/choose-a-model) - find the best model that fits on your Mac.
- [Network Boundary](/localcode/concepts/network-boundary) - learn what leaves your machine and when.
