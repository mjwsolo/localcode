---
title: Models
description: How localcode picks a model for your Mac, how the picker works, and what determines speed.
---

localcode has **no fixed default model**. When it starts, it checks your Mac's unified memory. It then recommends the most capable production-ready model whose weights fit in the available memory, and marks it with a star in the picker.

## The rule

Model weights must use about **55% of unified memory** or less. The rest is for the KV cache, activations, and macOS. localcode recommends the most capable model that fits. It never recommends experimental architectures automatically, but you can choose them yourself.

## Recommendations for each Mac

| Unified memory | Recommended | Quant | Weights |
| ---: | --- | --- | ---: |
| 16 GB | Gemma 4 12B | UD-Q4_K_XL | 7.37 GB |
| 24–48 GB | Qwen 3.6 35B-A3B | Q2 | 10.7 GB |
| 64 GB | Gemma 4 26B-A4B | Q8 | 28.0 GB |
| 96 GB+ | Qwen 3.6 35B-A3B | UD-Q8_K_XL | 38.5 GB |

## The model picker

The picker opens on first launch when no model is loaded. Type `/models` to open it at any time. Esc goes back one level.

**Level 1: models.** Every model in the catalog, shown as display name and maker. A count such as "2 on disk" tells you how many of its quants are already downloaded. The star marks the model recommended for this Mac's memory. Press Enter to open a model.

**Level 2: quants.** Every quant the model's Hugging Face repo ships, split into **Downloaded** and **Available to download**. Each row shows:

| Column | Meaning |
| --- | --- |
| Size | The GGUF file size in GB |
| Memory fit | `✓ fits`, `~ tight`, or `✗ too big` for this Mac |
| State | `Ready` on disk, or `Download` |

Press Enter on a `Ready` row to load it. The model server reloads on the same port and the session continues. Press Enter on a `Download` row to start the download; the row shows a live percentage (`⇣ 42%`). Press Enter on a downloading row to cancel it.

Nothing downloads without you choosing it and seeing its size. Downloads come from Hugging Face and happen once per quant.

A **Models folder** entry at the bottom of the picker shows where GGUFs are stored and lets you change it. The default is `~/.local/share/localcode/models`; `LOCALCODE_MODEL_DIR` overrides it. The older `LOCALCODE_MODELS_DIR` also works. See [Configuration](/localcode/reference/configuration).

## Starting with a model

`localcode --model TAG` starts with an already-downloaded model alias instead of opening the picker. It does not download anything.

## What determines speed

Three things matter, roughly in this order:

1. **Active parameters per token.** On Apple Silicon, memory bandwidth limits decoding speed. A Mixture-of-Experts model uses only a few billion parameters per token. It reads far fewer bytes per token than a dense model with the same total size.
2. **Memory bandwidth.** This varies much more between chip tiers than the number of cores. It directly affects decoding speed.
3. **KV cache size.** TurboQuant compression (`q8_0`-K + `turbo4`-V) keeps the cache small enough for long contexts to remain practical. See [Unified Memory](/localcode/concepts/unified-memory).

localcode does not run a benchmark on your machine and does not publish throughput figures. Real speed depends on your chip, context length, thermal state, and other running tasks.

## Hidden reasoning

Hidden reasoning ("thinking") is off by default for every model. The model server starts with `--reasoning off`. `/thinking` changes how thinking is displayed when a model produces it.

## Vision

After a vision-capable model loads, a hint says its image projector is available. `/vision` shows the projector's size and downloads it after you confirm. From then on that model can see images: drag or paste one into the prompt, or let the `read` tool open a PNG. The projector is per model.

## Next

- [Unified Memory](/localcode/concepts/unified-memory) - explains the memory budget.
