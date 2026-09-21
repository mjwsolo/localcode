---
title: Install
description: Install localcode, open a repo, pick a model, make one change.
---

```sh
pip install -U localcode
```

| | |
| --- | --- |
| Machine | Mac with Apple Silicon |
| macOS | 13 or newer |
| Unified memory | At least 16 GB |
| Python | 3.10 or newer |
| Disk | Space for one model - the smallest recommended GGUF is about 7.4 GB |

The wheel is built for `macosx_13_0_arm64` and ships two binaries: `llama-server` for inference and `localcode-ui` for the interface. Nothing else needs to be installed.

Older Macs are not supported. If launch fails with a `dyld` "Library not loaded" error, the Mac is below the minimum macOS. See [Error Codes](/localcode/reference/error-codes).

## Open your repo

```sh
cd ~/work/some-project
localcode
```

Choose a project whose tests already pass. localcode uses your repo's own checks as proof.

`localcode` opens the home screen. The previous 0.3 interface is still there: run `localcode --classic`, and read its docs through the version switcher in the header.

## Choose a model

With no model loaded, the model picker opens on top of the home screen. The first level lists every model in the catalog with its maker, how many of its quants are already on disk, and a star on the model recommended for your Mac's unified memory. Press Enter on a model to see every quant its Hugging Face repo ships, with the size in GB and whether it fits in memory. Enter on a row marked **Download** starts the download and shows a live percentage. Nothing downloads until you choose it and see its size.

![The localcode model picker: seven models, moving down the list and choosing one](/localcode/demo/step-2-choose-model.gif?v=a0c3cc9d)

When the download finishes, the included `llama-server` loads the model on a localhost port and the interface connects to it. Type `/models` at any time to open the picker again. Esc goes back one level.

Learn more in [Models](/localcode/start-here/choose-a-model).

## Start building

Enter your request in the prompt. Include the file name and the check to run.

![Entering a goal in the localcode prompt and pressing Enter](/localcode/demo/step-3-ask.gif?v=0523fe0c)

```text
> Implement the retry decorator in retry.py so every test in test_retry.py
  passes. Do not modify test_retry.py. Then run: pytest -q
```

## Watch it verify

The model reads the stub and tests. It writes the code, runs `pytest -q`, and reports what it checked. The first file edit and the first shell command each ask for permission; choose **always** to stop being asked for the rest of the session. See [Permissions](/localcode/start-here/permissions).

![localcode reading files, editing them, and then showing 5 passed in pytest](/localcode/demo/step-4-verify.gif?v=92c546ff)

<small>Qwen3.6-35B-A3B (IQ2_M) runs locally on a `127.0.0.1` port. The repository's
tests fail before the turn and pass after it.</small>

## Key commands

| Command | What it does |
| --- | --- |
| `/models` | Opens the model picker to download or switch models |
| `/status` | Shows the server, the current model, and the session |
| `/permissions` | Reviews what the agent may do without asking |
| `/new` | Starts a new session |
| `/sessions` | Switches to an earlier session |
| `/undo` | Undoes the previous message and its edits |
| `/help` | Lists every command and key |
| `/exit` | Quits |

Type `/` to open the command palette. Start a line with `!` to run a shell command yourself. See the full list: [Slash Commands](/localcode/reference/slash-commands).

## Next

- [Models](/localcode/start-here/choose-a-model) - find the best model that fits on your Mac.
- [Network Boundary](/localcode/concepts/network-boundary) - learn what leaves your machine and when.
