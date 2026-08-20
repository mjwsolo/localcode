<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/lockup-dark-v2.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/lockup-light-v2.png">
    <img alt="localcode" src="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/lockup-light-v2.png" width="360">
  </picture>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/localcode?style=flat-square&labelColor=171A1D&color=8AB4FF" alt="PyPI">
  <img src="https://img.shields.io/badge/license-Apache_2.0-8AB4FF?style=flat-square&labelColor=171A1D" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-8AB4FF?style=flat-square&labelColor=171A1D" alt="Python">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-8AB4FF?style=flat-square&labelColor=171A1D" alt="Platform">
</p>

<p align="center">
  <strong>A coding agent that runs one local model well on your Mac.</strong><br>
  llama.cpp + GGUF. No cloud inference, API keys, or account.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/demo/first-change.gif" alt="A real localcode turn: two reads, one edit, then pytest reporting 5 passed" width="900">
</p>

## What actually runs, and where

localcode serves **one GGUF model at a time** through a bundled `llama-server` on `127.0.0.1`. Its agent loop reads and edits files, runs your tests, and tells you what it verified.

localcode reads your Mac's unified memory at startup and picks the most
capable model whose weights fit inside about 55% of it, leaving room for the
KV cache and macOS. There is no fixed default.

| Unified memory | Auto-selected model | Weights | Models that fit |
| --- | --- | ---: | ---: |
| 16 GB | Gemma 4 12B (Q4) | 7.37 GB | 1 |
| 24 GB | Qwen 3.6 35B-A3B (Q2) | 10.7 GB | 3 |
| 32 GB | Qwen 3.6 35B-A3B (Q2) | 10.7 GB | 5 |
| 48 GB | Qwen 3.6 35B-A3B (Q2) | 10.7 GB | 8 |
| 64 GB | Gemma 4 26B-A4B (Q8) | 28 GB | 9 |
| 96 GB+ | Qwen 3.6 35B-A3B (Q8) | 38.5 GB | 10 |

Throughput depends on the model, the quant and the machine, so the only
numbers published here are ones actually measured. On the development
machine — **MacBook Pro, M5 Max, 128 GB, macOS 26.5** — running
Qwen3.6-35B-A3B (IQ2_M) at a 131072-token context:

| | |
|---|---|
| Generation | ~89 tokens/sec |
| Prompt eval | ~1174 tokens/sec |
| A 4-tool-call task, warm | 12–15 s end to end |

Numbers for other Macs have not been measured yet, so none are published.

Inference is local by default. Some features *do* use the network: model downloads, the `web_search` / `web_fetch` tools, and MCP servers.

## Install

```bash
pip install -U localcode      # or: uv pip install -U localcode
```

## Run

```bash
cd your-project
localcode
```

## Docs

Read the docs at [mjwsolo.github.io/localcode](https://mjwsolo.github.io/localcode/).

## What it does

- **Reads and edits files** — understands your codebase, makes small and precise edits, and refuses destructive overwrites
- **Runs commands** — runs tests, builds, Git, and shell commands; detects long-running servers and moves them to the background
- **Searches code** — searches by filename pattern, content with grep, or directory structure
- **Builds and launches apps** — detects `package.json`, `pyproject.toml`, or static apps; picks a free port; then starts and checks the process
- **Tracks tasks across turns** — keeps the task state, stage (scaffolding → implementing → verifying), and goal between user messages
- **Adaptive thinking** — uses reasoning for planning and debugging, but skips it for routine codegen
- **Uses tools automatically** — the model chooses its own tools

```
> build me a Flask app for studying music theory with quizzes
```

localcode works out the goal, sets up the project, writes the files, runs `pip install`, starts the server, opens it in your browser, and checks that it responds. It does all of this locally.

## Why local?

We want powerful, personal AI to be available to everyone, on any device and in any place. That means true local-first AI. localcode is the first step toward that goal.

## Requirements

- **Mac with Apple Silicon**
- **16 GB RAM** minimum
- **Python 3.10+**
- **~12 GB free disk** (10 GB model + server)

## Models

When it starts, localcode recommends the best model for **your Mac's RAM**. There is no fixed default. You can choose any model below, or select a different quant in the model picker.

| Model | Size (quant) | Active params | Min RAM |
| --- | ---: | --- | ---: |
| Gemma 4 12B | 7.4 GB (Q4) | 12B (dense) | 16 GB |
| Gemma 4 26B-A4B | 11.2 GB (Q3) | 3.8B (8/128 experts) | 24 GB |
| Qwen 3.6 35B-A3B | 10.7 GB (Q2) | 3.0B (8+1/256) | 24 GB |
| Qwen 3.8 27B | 17.9 GB (Q4) | 27B (dense) | 32 GB |
| Muse Glimmer 30B | 15.9 GB (Q4) | 30B (dense, multimodal) | 32 GB |
| DiffusionGemma 26B-A4B | 15.7 GB (Q4) | 4B (diffusion MoE) | 32 GB |
| North-Mini-Code 30B-A3B | 17.9 GB (Q4) | 3B (30B MoE) | 36 GB |
| Gemma 4 12B (full) | 23.8 GB (BF16) | 12B (dense) | 48 GB |
| Gemma 4 26B-A4B | 28 GB (Q8) | 3.8B (8/128 experts) | 64 GB |
| Qwen 3.6 35B-A3B | 38.5 GB (Q8) | 3.0B (8+1/256) | 96 GB |

## How localcode works

localcode uses a custom fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with **TurboQuant KV cache compression**. TurboQuant comes from Google's ICLR 2026 paper, and we added it to llama.cpp for Apple Silicon. It makes the KV cache 3.8× smaller. This lets a 16 GB MacBook fit a 32K context in 355 MiB.

localcode chooses a model based on **your Mac's RAM**. There is no fixed default. It ranges from Gemma 4 12B on 16 GB to Qwen 3.6 35B-A3B on 64 GB+. The recommended models use Mixture-of-Experts. Only about 3.8 B parameters are active for each token. This makes about 27 tok/s possible on a laptop.

Under the hood:

- **TurboQuant KV cache** — uses asymmetric q8\_0-K + turbo4-V quantization and is 3.8× smaller than f16
- **Multi-region mmap patch** — fixes a Metal OOM crash caused by llama.cpp's loader putting the whole GGUF file into one Metal buffer
- **GPU memory unlock** — asks to raise `iogpu.wired_limit_mb` automatically so the model can fully use Metal
- **Agent loop** — routes goals by type (build / edit / run / chat), keeps task state, requires evidence before finishing, and has recovery modes for common small-model failures

## Sponsors

To sponsor localcode, [reach out](https://github.com/mjwsolo/localcode).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
