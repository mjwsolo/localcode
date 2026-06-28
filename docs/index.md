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
  <img src="https://img.shields.io/badge/python-3.10+-3776ab?style=flat-square" alt="Python">
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
- **Python 3.10+**
- **~12 GB free disk** (10 GB model + server)

### Tested hardware

LocalCode is early software. Hardware support is expected to broaden, but only the configuration below has been tested by the maintainers so far.

| Mac | Memory | Status | Notes |
| --- | ---: | --- | --- |
| M4 MacBook | 16 GB | Tested | Primary development and validation machine |
| M1/M2/M3 Apple Silicon | 16 GB+ | Not yet tested | Expected to work, but needs validation |
| M4 Apple Silicon | 24 GB+ | Not yet tested | Expected to support larger contexts, but needs validation |
| Intel Mac | Any | Not supported | LocalCode targets Apple Silicon |

## Models

On launch, LocalCode recommends the best model for **your Mac's RAM** — there's no fixed default. You can pick any of these (or a different quant) in the model picker.

| Model | Size (quant) | Active params | Min RAM | Architecture |
| --- | ---: | --- | ---: | --- |
| Gemma 4 12B | 7.4 GB (Q4) | 12B (dense) | 16 GB | gemma4-iswa |
| Gemma 4 26B-A4B | 11.2 GB (Q3) | 3.8B (8/128 experts) | 24 GB | gemma4-iswa |
| Qwen 3.6 35B-A3B | 10.7 GB (Q2) | 3.0B (8+1/256) | 24 GB | qwen35moe |
| DiffusionGemma 26B-A4B † | 15.7 GB (Q4) | 4B (diffusion MoE) | 32 GB | diffusion_gemma |
| North-Mini-Code 30B-A3B † | 17.9 GB (Q4) | 3B (30B MoE) | 36 GB | cohere2_moe |
| Gemma 4 12B (full) | 23.8 GB (BF16) | 12B (dense) | 48 GB | gemma4-iswa |
| Gemma 4 26B-A4B | 28 GB (Q8) | 3.8B (8/128 experts) | 64 GB | gemma4-iswa |
| Qwen 3.6 35B-A3B | 38.5 GB (Q8) | 3.0B (8+1/256) | 96 GB | qwen35moe |

*Min RAM* is the threshold for auto-recommendation (weights ≤ ~55% of unified memory, leaving room for KV cache + OS); you can still pick a heavier model manually. **†** experimental — pickable but **not** auto-recommended (DiffusionGemma needs a separate runner; `cohere2_moe` is unvalidated on this stack).

## How LocalCode works

LocalCode runs a custom [llama.cpp](https://github.com/ggerganov/llama.cpp) fork with **TurboQuant KV cache compression** — a technique from Google's ICLR 2026 paper that we patched into llama.cpp for Apple Silicon. This compresses the KV cache 3.8× — fitting 32K context in 355 MiB on a 16 GB MacBook.

LocalCode picks a model based on **your Mac's RAM** — there's no fixed default. It scales from Gemma 4 12B on 16 GB up to Qwen 3.6 35B-A3B on 64 GB+. The recommended models are Mixture-of-Experts — only ~3.8 B parameters active per token — which is what makes ~27 tok/s possible on a laptop.

Under the hood:

- **TurboQuant KV cache** — asymmetric q8\_0-K + turbo4-V quantization, 3.8× compression vs. f16
- **Multi-region mmap patch** — fixes a Metal OOM crash where llama.cpp's loader spanned the entire GGUF file into one Metal buffer
- **GPU memory unlock** — auto-prompts to raise `iogpu.wired_limit_mb` for full Metal offload
- **Agent loop** — goal-typed routing (build / edit / run / chat) with task state, evidence-driven completion, and recovery modes for small-model failure patterns

## Sponsors

If you'd like to sponsor LocalCode, [reach out](https://github.com/mjwsolo/localcode).

## Contributing

See [CONTRIBUTING.md](https://github.com/mjwsolo/localcode/blob/main/CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](https://github.com/mjwsolo/localcode/blob/main/LICENSE).
