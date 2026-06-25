<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/light.png">
    <img alt="LocalCode" src="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/light.png" width="480">
  </picture>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/localcode?style=flat-square&color=7c4dff" alt="PyPI">
  <img src="https://img.shields.io/badge/license-Apache_2.0-4caf50?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/python-3.11+-3776ab?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-999999?style=flat-square" alt="Platform">
</p>

<p align="center">
  <strong>High-performance AI coding on consumer hardware.</strong><br>
  No cloud, no API keys, no data leaving your machine.
</p>

> ⚠️ **Alpha software.** Active development; expect rough edges, breaking changes between versions, and bugs. Issues and feedback welcome.

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

## Docs

Docs are published at [mjwsolo.github.io/localcode](https://mjwsolo.github.io/localcode/).

## What it does

- **Reads and edits files** — understands your codebase, makes surgical edits, refuses destructive overwrites
- **Runs commands** — tests, builds, git, shell; auto-detects long-running servers and backgrounds them
- **Searches code** — by filename pattern, content (grep), or directory structure
- **Builds and launches apps** — detects `package.json` / `pyproject.toml` / static, picks a free port, starts and verifies the process
- **Tracks tasks across turns** — task state, stage (scaffolding → implementing → verifying), and goal carry between user messages
- **Adaptive thinking** — uses reasoning for planning and debugging, skips it for routine codegen
- **Uses tools automatically** — the model picks its own tools

```
> build me a Flask app for studying music theory with quizzes
```

LocalCode infers the goal, scaffolds the project, writes the files, runs `pip install`, launches the server, opens it in your browser, and verifies it responds — all locally.

## Why local?

We are building for a world of truly democratized AI — where everyone has access to powerful, personalized AI on any device, in any location. True local-first AI. LocalCode is the first step toward that vision.

## Requirements

- **Mac with Apple Silicon**
- **16 GB RAM** minimum
- **Python 3.11+**
- **~12 GB free disk** (10 GB model + server)

### Tested hardware

LocalCode is early software. Hardware support is expected to broaden, but only the configuration below has been tested by the maintainers so far.

| Mac | Memory | Status | Notes |
| --- | ---: | --- | --- |
| M4 MacBook | 16 GB | Tested | Primary development and validation machine |
| M1/M2/M3 Apple Silicon | 16 GB+ | Not yet tested | Expected to work, but needs validation |
| M4 Apple Silicon | 24 GB+ | Not yet tested | Expected to support larger contexts, but needs validation |
| Intel Mac | Any | Not supported | LocalCode targets Apple Silicon |

## How LocalCode works

LocalCode runs a custom [llama.cpp](https://github.com/ggerganov/llama.cpp) fork with **TurboQuant KV cache compression** — a technique from Google's ICLR 2026 paper that we patched into llama.cpp for Apple Silicon. This compresses the KV cache 3.8× — fitting 32K context in 355 MiB on a 16 GB MacBook.

The default model (**Gemma 4 26B-A4B**) is a Mixture-of-Experts architecture — 25.2 B total parameters but only 3.8 B active per token. That's what makes ~27 tok/s possible on a laptop.

Under the hood:

- **TurboQuant KV cache** — asymmetric q8\_0-K + turbo4-V quantization, 3.8× compression vs. f16
- **Multi-region mmap patch** — fixes a Metal OOM crash where llama.cpp's loader spanned the entire GGUF file into one Metal buffer
- **GPU memory unlock** — auto-prompts to raise `iogpu.wired_limit_mb` for full Metal offload
- **Agent loop** — phased CREATE/EDIT/CHAT/RUN/SEARCH routing with task state, evidence-driven completion, and recovery modes for small-model failure patterns

## Sponsors

If you'd like to sponsor LocalCode, [reach out](https://github.com/mjwsolo/localcode).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
