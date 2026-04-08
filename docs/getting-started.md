# Getting Started

Let's get LocalCode running on your machine.

## Requirements

- **Mac with Apple Silicon** (M1, M2, M3, or M4)
- **16GB RAM** minimum (24GB+ recommended for bigger context)
- **Python 3.11+**
- **~12GB free disk** for the model and server binary

!!! note
    cmake and Xcode Command Line Tools are installed automatically if missing. Ollama is detected and used as a fallback if available.

## Install

=== "pip (recommended)"

    ```bash
    git clone https://github.com/mjwsolo/localcode.git
    cd localcode
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

=== "pipx"

    ```bash
    pipx install git+https://github.com/mjwsolo/localcode.git
    ```

## Run

```bash
localcode
```

That's it. First launch handles everything:

1. **Builds** the TurboQuant inference server from source (~3 min)
2. **Downloads** Gemma 4 26B IQ3_S (~10GB, one time)
3. **Unlocks** GPU memory on 16GB Macs (asks for sudo, resets on reboot)

After the first run, startup takes ~15 seconds.

## Pick a mode

```
  Select a mode:

  1. Fast         27 tok/s  32K context
  2. Reasoning    26 tok/s  32K context
```

**Fast** — best for straightforward coding tasks. No thinking overhead.

**Reasoning** — enables thinking mode. The model reasons through problems before responding. Great for complex multi-step tasks.

!!! tip
    Start with Fast. Type `/switch` anytime to toggle to Reasoning when you hit something that needs deeper thought.

## Try it out

Good first things to try:

```
read pyproject.toml and summarize what this project does
```

```
find all Python files that import os
```

```
write a hello world script and run it
```

The model will pick the right tools on its own — reading files, running commands, searching code.

## Performance

| Metric | 16GB Mac | 24GB+ Mac |
|--------|----------|-----------|
| Decode speed | 27 tok/s | 27+ tok/s |
| Time to first token | 270ms | 270ms |
| Prompt eval | 87 tok/s | 87+ tok/s |
| Context window | 32K | 48K–128K |
| KV cache | 355 MiB | scales up |

## What's next

- [First Run](first-run.md) — detailed walkthrough of the first launch
- [Tools](tools.md) — what the model can do
- [Configuration](configuration.md) — tune settings
- [Troubleshooting](troubleshooting.md) — if something goes wrong
