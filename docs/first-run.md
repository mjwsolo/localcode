# First Run

Here's what happens the first time you launch LocalCode.

## Step 1: Build the server

LocalCode uses a custom llama.cpp fork with TurboQuant KV cache support. The first run compiles it from source:

```
Building TurboQuant server...  ████████████████████  3:12
```

This needs cmake and a C++ compiler. Both are installed automatically if missing (via Xcode Command Line Tools on Mac).

!!! note
    The build takes 2-4 minutes depending on your machine. This only happens once.

## Step 2: Download the model

Gemma 4 26B IQ3_S (~10GB) is downloaded from Hugging Face:

```
Downloading model...  ████████████████████  10.4 GB
```

The model is cached at `~/.gem/models/` so it's only downloaded once.

## Step 3: GPU memory unlock

On 16GB Macs, Apple Silicon limits GPU memory to ~11GB by default. Our model + KV cache needs more. LocalCode asks for sudo to raise the limit:

```
GPU memory unlock needed for full speed.
This is safe — resets on reboot, no permanent changes.
Password: ▊
```

```bash
sudo sysctl iogpu.wired_limit_mb=14336
```

!!! info "What this does"
    Raises the Metal GPU working set from ~11GB to 14GB. This enables full GPU offload (27 tok/s) instead of falling back to CPU (18 tok/s). Resets automatically on reboot.

!!! tip "24GB+ Macs"
    If you have 24GB or more RAM, this step is skipped. You already have enough GPU headroom.

## Step 4: Mode selection

```
  Select a mode:

  1. Fast         27 tok/s  32K context
  2. Reasoning    26 tok/s  32K context
```

Pick one and you're in. The server starts, the model loads, and you're talking to a local 26B coding agent.

!!! tip
    You can switch modes anytime with `/switch` — no restart needed.

## Good first commands

| Try this | What happens |
|----------|-------------|
| `read pyproject.toml` | Model reads and summarizes the file |
| `find all test files` | Model searches your repo |
| `/switch` | Toggle between fast and reasoning mode |
| `/help` | Shows all available commands |
| `/status` | Shows runtime info — model, mode, repo |

## Next steps

- [Tools](tools.md) — what the model can do
- [Commands](commands.md) — all available slash commands
- [Configuration](configuration.md) — runtime settings
