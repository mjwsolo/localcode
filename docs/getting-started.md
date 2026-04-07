# Getting Started

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM minimum
- Python 3.10+
- ~12GB free disk space

## Installation

```bash
# Clone the repo
git clone https://github.com/mjwsolo/localcode.git
cd localcode

# Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Build the inference server
cd llama-cpp-turboquant
./BUILD.sh
cd ..

# Download the model via Ollama
brew install ollama
ollama pull gemma4:26b-a4b
```

## First Run

```bash
localcode
```

On first launch:
1. You'll be asked to allow GPU memory unlock (one-time per boot, resets on reboot)
2. Select your mode (Fast or Reasoning)
3. The server starts automatically
4. Start coding!

## Quick Test

```
> hi
Hello! How can I help you today?

> make a pong game using pygame
  create: full generation path
  · generating code...
  ✓ write — pong.py (150 lines)
  ✓ verify — syntax OK
  ✓ deps — all imports OK

> run it for me
  ● bash python pong.py
```

## Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history |
| `/undo` | Revert last file change |
| `/verify` | Run syntax check on last file |
| `/quit` | Exit |
