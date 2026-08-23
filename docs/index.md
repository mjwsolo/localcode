<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/light.png">
    <img alt="localcode" src="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/logo/light.png" width="480">
  </picture>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/localcode?style=flat-square&labelColor=171A1D&color=8AB4FF" alt="PyPI">
  <img src="https://img.shields.io/badge/license-Apache_2.0-8AB4FF?style=flat-square&labelColor=171A1D" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-8AB4FF?style=flat-square&labelColor=171A1D" alt="Python">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-8AB4FF?style=flat-square&labelColor=171A1D" alt="Platform">
</p>

<p align="center">
  <strong>A coding agent that runs a local model on your Mac.</strong><br>
  No cloud inference, no API key, no account.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mjwsolo/localcode/main/docs/assets/demo/first-change.gif" alt="A localcode turn: two reads, one edit, then pytest reporting 5 passed" width="900">
</p>

> **Alpha software.** Expect rough edges and breaking changes between versions. Issues and feedback are welcome.

localcode runs an open-weight model on your Mac and uses it to read, edit and test your code. Your prompts and your files stay on your machine. The only thing it downloads is the model weights, once per model.

## Install

```bash
pip install -U localcode      # or: uv pip install -U localcode
```

The inference server ships inside the package. Nothing is compiled or cloned on your machine.

## Run

```bash
cd your-project
localcode
```

On first launch localcode recommends a model for your Mac's memory. Pick one, wait for the download, and start typing.

```
> Implement the retry decorator in retry.py so every test in test_retry.py passes. Then run: pytest -q
```

Every error localcode shows has a code. Look it up in the [Errors reference](ERRORS.md).

## What it does

- Reads and edits files in your project
- Runs your tests, builds, Git and shell commands, and asks before anything risky
- Searches code by name, content or structure
- Scaffolds and launches apps, then checks that they respond
- Remembers the task across messages
- Undoes its own changes with `/undo`

## Requirements

- Mac with Apple Silicon
- 16 GB unified memory or more
- Python 3.10 or newer
- About 12 GB of free disk for the smallest model

## Models

localcode recommends a model by your Mac's memory and marks it with a star. You choose; nothing is selected for you. Every model runs on binaries shipped in the package.

| Model | Weights | Quant | Active params | Min RAM |
| --- | ---: | --- | --- | ---: |
| Gemma 4 12B | 7.4 GB | UD-Q4_K_XL | 12B (dense) | 16 GB |
| Qwen 3.6 35B-A3B | 10.7 GB | UD-IQ2_M | 3.0B (MoE) | 24 GB |
| Gemma 4 26B-A4B | 11.2 GB | UD-IQ3_S | 3.8B (MoE) | 24 GB |
| DiffusionGemma 26B-A4B | 15.7 GB | Q4_K_M | 4B (diffusion MoE) | 32 GB |
| Muse Glimmer 30B | 15.9 GB | UD-Q4_K_XL | 30B (dense, vision) | 32 GB |
| Qwen 3.8 27B | 17.9 GB | UD-Q4_K_XL | 27B (dense) | 36 GB |
| North-Mini-Code 30B-A3B | 17.9 GB | UD-Q4_K_M | 3B (MoE) | 36 GB |
| Gemma 4 12B (full) | 23.8 GB | BF16 | 12B (dense) | 48 GB |
| Gemma 4 26B-A4B | 28.0 GB | UD-Q8_K_XL | 3.8B (MoE) | 64 GB |
| Qwen 3.6 35B-A3B | 38.5 GB | UD-Q8_K_XL | 3.0B (MoE) | 96 GB |

Min RAM is the memory at which localcode will recommend the model. You can pick a heavier one by hand. Every model runs on the same bundled server. DiffusionGemma is a research model that is never recommended automatically; its replies arrive a block at a time rather than word by word.

Measured on a MacBook Pro (M5 Max, 128 GB) with Qwen 3.6 35B-A3B UD-IQ2_M at a 131072-token context: about 89 tokens/s generation, about 1174 tokens/s prompt processing, and 12 to 15 seconds for a typical four-tool-call task.

## Network

Inference is local. Three features use the network: model downloads, the `web_search` and `web_fetch` tools, and any MCP servers you add. See [Network Boundary](https://mjwsolo.github.io/localcode/concepts/network-boundary/) for the full list.

## Why local?

Powerful, personal AI should work for everyone, on any device, anywhere. That means running it locally. localcode is a first step.

## Sponsors

To sponsor localcode, [reach out](https://github.com/mjwsolo/localcode).

## Contributing

See [CONTRIBUTING.md](https://github.com/mjwsolo/localcode/blob/main/CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](https://github.com/mjwsolo/localcode/blob/main/LICENSE).
