<h1 align="center">🏠 LocalCode</h1>

<p align="center">
  <img src="https://img.shields.io/badge/build-passing-4caf50?style=flat-square" alt="Build">
  <img src="https://img.shields.io/badge/release-v0.1.1-7c4dff?style=flat-square" alt="Release">
  <img src="https://img.shields.io/badge/license-Apache--2.0-4caf50?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/python-3.11+-3776ab?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-999999?style=flat-square" alt="Platform">
</p>

<p align="center">
  <strong>High-performance AI coding on consumer hardware.</strong><br>
  No cloud, no API keys, no data leaving your machine.
</p>

## Install

```bash
pip install localcode
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
- **Fast mode** - for routine coding tasks
- **Reasoning mode** - deep thinking for complex multi-step problems
- **Uses tools automatically** - the model picks its own tools

```
> refactor the auth module to use JWT and make sure the tests pass
```

LocalCode reads the files, plans the refactor, edits the code, runs the tests, and fixes failures - all locally.

## Why local?

We are building for a world of truly democratized AI - where everyone has access to powerful, personalized, prompt AI anywhere, on any device, and in any location. True empowered local-first AI. LocalCode is the first step toward that vision.

### How LocalCode compares

| | LocalCode | Claude Code | OpenCode | Codex CLI |
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

## How LocalCode works

LocalCode runs a custom [llama.cpp](https://github.com/ggerganov/llama.cpp) fork with **TurboQuant KV cache compression** - a technique from Google's ICLR 2026 paper that we patched into llama.cpp for Apple Silicon. This compresses the KV cache 3.8x, fitting 32K context in 355 MiB on a 16GB MacBook.

The model (**Gemma 4 26B-A4B**) is a Mixture-of-Experts architecture - 25.2B total parameters but only 3.8B active per token. That's what makes 27 tok/s possible on a laptop.

## Sponsors

If you'd like to sponsor LocalCode, [reach out](https://github.com/mjwsolo/localcode).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
