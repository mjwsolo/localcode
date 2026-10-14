<p align="center">
  <img alt="localcode" src="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/lockup-dark.svg" width="360">
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/localcode?style=flat-square&labelColor=171A1D&color=8AB4FF" alt="PyPI">
  <img src="https://img.shields.io/badge/license-Apache_2.0-8AB4FF?style=flat-square&labelColor=171A1D" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-8AB4FF?style=flat-square&labelColor=171A1D" alt="Python">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-8AB4FF?style=flat-square&labelColor=171A1D" alt="Platform">
</p>

<p align="center">
  <strong>Agentic coding. Local models. On your Mac.</strong>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/demo/first-change.gif" alt="A localcode turn: it reads the stub and test, edits the file, then pytest reports 1 passed" width="900">
</p>

## Install

```bash
pip install -U localcode      # or: uv pip install -U localcode
```

## Run

```bash
cd your-project
localcode
```

```
> Implement the retry decorator in retry.py so every test in test_retry.py passes. Then run: pytest -q
```

To use the running model from another local app, open `/server` or run `localcode api` for its
OpenAI compatible endpoint. See the [Local API guide](https://mjwsolo.github.io/localcode/guides/local-api/).

## What it does

- Reads and edits files in your project
- Runs your tests, builds, Git and shell commands inside the project on its own, and asks before touching anything outside it
- Searches code by name, content or structure
- Scaffolds and launches apps, then checks that they respond
- Remembers the task across messages

## Requirements

- Mac with Apple Silicon, macOS 13 or newer
- 16 GB unified memory or more
- Python 3.10 or newer
- Disk space for LocalCode and your chosen model (download sizes below)

## Models

Choose from **10 models and 100+ quant options** for coding, image input and typed decisions. Models download from Hugging Face. Downloads run in parallel and can resume if interrupted. Switch with `/models`.

| Model | Smallest download | Largest download |
| --- | ---: | ---: |
| [Gemma 4 12B](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) | 4.2 GB | 23.8 GB |
| [Qwen 3.8 27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 6.2 GB | 31.5 GB |
| [Underdog Saluki 27B 1.0](https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0) | 7.9 GB | 7.9 GB |
| [North-Mini-Code 30B-A3B](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF) | 9.4 GB | 33.2 GB |
| [Gemma 4 26B-A4B](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | 9.9 GB | 27.6 GB |
| [Qwen 3.6 35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | 10.0 GB | 38.5 GB |
| [Muse Glimmer 30B](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) | 10.7 GB | 32.3 GB |
| [DiffusionGemma 26B-A4B](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | 16.8 GB | 50.5 GB |
| [OpenJev 27B (decisions)](https://huggingface.co/ggml-org/OpenJev-GGUF) | 19.4 GB | 53.8 GB |
| [Kolibri 1 78B-A3B (experimental)](https://huggingface.co/Hob-forge/Kolibri-1-GGUF) | 28.6 GB | 47.5 GB |

## Network

Inference is local. Three features use the network: model downloads, the `web_search` and `web_fetch` tools, and any MCP servers you add. See [Network Boundary](https://mjwsolo.github.io/localcode/concepts/network-boundary/) for the full list.

## Why local?

Powerful, personal AI should work for everyone, on any device, anywhere. That means running it locally. localcode is a first step.

## Sponsors

To sponsor localcode, [reach out](https://github.com/mjwsolo/localcode).

Installation diagnostics: `localcode doctor`. See [compatibility tests and local hardware verification](docs/src/content/docs/development/verification.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
