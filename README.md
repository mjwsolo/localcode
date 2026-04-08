<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo.png">
    <source media="(prefers-color-scheme: light)" srcset="logo-light.png">
    <img src="logo.png" alt="localcode" width="500">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/mjwsolo/localcode/actions"><img src="https://img.shields.io/github/actions/workflow/status/mjwsolo/localcode/ci.yml?label=build&style=flat-square" alt="Build"></a>
  <a href="https://github.com/mjwsolo/localcode/releases"><img src="https://img.shields.io/github/v/release/mjwsolo/localcode?style=flat-square&color=7c4dff&label=release" alt="Release"></a>
  <a href="https://github.com/mjwsolo/localcode/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mjwsolo/localcode?style=flat-square&color=4caf50&label=license" alt="License"></a>
  <a href="https://github.com/mjwsolo/localcode"><img src="https://img.shields.io/github/stars/mjwsolo/localcode?style=flat-square&color=f5c542&label=stars" alt="Stars"></a>
  <img src="https://img.shields.io/badge/python-3.11+-3776ab?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-999999?style=flat-square" alt="Platform">
</p>

<p align="center">
  <strong>The local AI coding agent.</strong> High-performance agentic coding on consumer hardware. No cloud, no API keys, no data leaving your machine.
</p>

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

- **Reads and edits files** - understands your codebase, makes surgical edits
- **Runs commands** - tests, builds, git, shell
- **Searches code** - by pattern, content, or semantic meaning
- **Fast mode** - 27 tok/s for routine coding tasks
- **Reasoning mode** - deep thinking for complex multi-step problems
- **Uses tools automatically** - the model picks its own tools

```
> refactor the auth module to use JWT and make sure the tests pass
```

localcode reads the files, plans the refactor, edits the code, runs the tests, and fixes failures - all locally.

## Why local?

We are building for a world of truly democratized AI - where everyone has access to powerful, personalized, prompt AI anywhere, on any device, and in any location. True empowered local-first AI. localcode is the first step toward that vision.

### How localcode compares

| | localcode | Claude Code | OpenCode | Codex CLI |
|--|-----------|-------------|----------|-----------|
| **Runtime** | 100% on-device | Cloud (Anthropic API) | Cloud (any provider) | Cloud (OpenAI API) |
| **Privacy** | Code never leaves your machine | Code sent to Anthropic | Code sent to provider | Code sent to OpenAI |
| **Cost** | Free forever | $100+/mo (Max) or API credits | API credits (varies) | Free (included with ChatGPT) |
| **Internet required** | No | Yes | Yes | Yes |


## Requirements

- **Mac with Apple Silicon** (M1/M2/M3/M4)
- **16GB RAM** minimum
- **Python 3.11+**
- **~12GB free disk**

## Key commands

| Command | What it does |
|---------|-------------|
| `/switch` | Toggle between fast (27 tok/s) and reasoning (26 tok/s) mode |

## How localcode works

localcode runs a custom [llama.cpp](https://github.com/ggerganov/llama.cpp) fork with **TurboQuant KV cache compression** - a technique from Google's ICLR 2026 paper that we patched into llama.cpp for Apple Silicon. This compresses the KV cache 3.8x, fitting 32K context in 355 MiB on a 16GB MacBook.

The model (**Gemma 4 26B-A4B**) is a Mixture-of-Experts architecture - 25.2B total parameters but only 3.8B active per token. That's what makes 27 tok/s possible on a laptop.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
