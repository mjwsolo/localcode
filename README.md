<p align="center">
  <img src="logo.png" alt="LocalCode" width="500">
</p>

<p align="center">
  <strong>The local AI coding agent.</strong> Runs Gemma 4 26B entirely on your Mac. No cloud, no API keys, no data leaving your laptop.
</p>

We are building for a world of truly democratized AI, where everyone has access to powerful, personalized, prompt AI anywhere, on any device, and in any location. True empowered local-first AI. LOCALcode is the first step toward that vision.

LocalCode is an open-source terminal coding assistant that reads your codebase, edits files, runs commands, and searches your code — powered by a 26B parameter model running locally at 27 tokens/second.

## Install

```bash
pip install localcode
```

Or from source:

```bash
pip install git+https://github.com/mjwsolo/localcode.git
```

## Run

```bash
cd your-project
localcode
```

That's it. First launch builds the inference server and downloads the model (~5 min, one time). After that, startup is ~15 seconds.

## What it does

- **Reads and edits files** — understands your codebase, makes surgical edits
- **Runs commands** — tests, builds, git, shell
- **Searches code** — by pattern, content, or semantic meaning
- **Thinks through hard problems** — reasoning mode for complex multi-step tasks
- **Uses tools automatically** — the model picks the right tool for the job

```
> refactor the auth module to use JWT and make sure the tests pass
```

LocalCode reads the files, plans the refactor, edits the code, runs the tests, and fixes failures — all locally.

## Why local?

| | LocalCode | Cloud AI |
|--|-----------|----------|
| Privacy | 100% local | Code sent to servers |
| Cost | Free forever | $20+/month |
| Offline | Works anywhere | Needs internet |
| Speed | 27 tok/s | ~50 tok/s |
| Context | 32K tokens | 128K+ |

Your code stays on your machine. No telemetry, no data collection, no API keys.

## Requirements

- **Mac with Apple Silicon** (M1/M2/M3/M4)
- **16GB RAM** minimum
- **Python 3.11+**
- **~12GB free disk**

## Key commands

| Command | What it does |
|---------|-------------|
| `/switch` | Toggle between fast (27 tok/s) and reasoning (26 tok/s) mode |
| `/help` | Show all commands |
| `/status` | Runtime info |
| `/undo` | Revert last change |

## How it works

LocalCode runs a custom [llama.cpp](https://github.com/ggerganov/llama.cpp) fork with **TurboQuant KV cache compression** — a technique from Google's ICLR 2026 paper that we patched into llama.cpp for Apple Silicon. This compresses the KV cache 3.8x, fitting 32K context in 355 MiB on a 16GB MacBook.

The model (**Gemma 4 26B-A4B**) is a Mixture-of-Experts architecture — 25.2B total parameters but only 3.8B active per token. That's what makes 27 tok/s possible on a laptop.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
