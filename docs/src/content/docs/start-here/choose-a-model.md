---
title: Models
description: How localcode picks a model for your Mac, how the picker works, and what determines speed.
---

localcode has **no fixed default model**. When it starts, it checks your Mac's unified memory. It then recommends the most capable production-ready model whose weights fit in the available memory, and marks it with a star in the picker.

## The rule

Model weights must use about **55% of unified memory** or less. The rest is for the KV cache, activations, and macOS. localcode recommends the most capable model that fits. It never recommends experimental or opt-in models automatically, but you can choose them yourself.

## Recommendations for each Mac

| Unified memory | Recommended | Quant | Weights |
| ---: | --- | --- | ---: |
| 16 GB | Gemma 4 12B | UD-Q4_K_XL | 7.37 GB |
| 24–48 GB | Qwen 3.6 35B-A3B | Q2 | 10.7 GB |
| 64 GB | Gemma 4 26B-A4B | Q8 | 28.0 GB |
| 96 GB+ | Qwen 3.6 35B-A3B | UD-Q8_K_XL | 38.5 GB |

Other models in the picker are choices you make yourself:

- **Qwen 3.8 27B** - a dense 27B (about 18 GB at Q4, 31.5 GB at Q8) with a built-in drafter. Comfortable on 48 GB and up.
- **Gemma 4 12B and 26B-A4B, Qwen 3.6 35B-A3B** - the defaults above, in other quants.
- **Flint 40B-A2.5B** - a coding-agent MoE with about 2.5B active parameters: 25 GB at Q4 (48 GB Macs and up) and 42 GB at Q8 (96 GB and up). Opt-in. It reasons on every step and has no off switch, so its thinking stays hidden and agent tasks run slower than on Qwen 3.6.
- **North-Mini-Code, Muse Glimmer, Underdog Saluki, DiffusionGemma, Kolibri** - additional or experimental models.
- **OpenJev** - a decision model, not a coding model. See [Chat and Decisions](/localcode/guides/decisions).

## The model picker

With no model loaded, the picker opens when you send your first message. Type `/models` to open it at any time. Esc goes back one level.

**Level 1: models.** Every model in the catalog, shown as display name and maker. A count such as "2 on disk" tells you how many of its quants are already downloaded. The star marks the model recommended for this Mac's memory. Press Enter to open a model.

**Level 2: quants.** Every quant the model's Hugging Face repo ships, split into **Downloaded** and **Available to download**. Each row shows:

| Column | Meaning |
| --- | --- |
| Size | The GGUF file size in GB |
| Memory fit | `✓ fits`, `~ tight`, or `✗ too big` for this Mac |
| State | `Ready` on disk, or `Download` |

The list opens on a quant that is already downloaded, or on the recommended quant when none is. Press Enter on a `Ready` row to load it. The model server reloads on the same port and the session continues. Press Enter on a `Download` row to start the download; the row shows a live percentage (`⇣ 42%`). Press Enter on a downloading row to cancel it.

Downloads use parallel requests and save verified chunks. After cancelling, losing your connection, or restarting LocalCode, select the same quant again to resume. Resume needs range support and a stable server ETag; otherwise LocalCode starts a fresh transfer. Completed chunks are checked before reuse, and the finished file is verified against its catalog checksum when one is available.

Partial downloads stay in the Models folder as `<filename>.part` and `<filename>.resume.json`. To reclaim that space, cancel the download, wait for it to stop, then delete those two files.

Nothing downloads without you choosing it and seeing its size. Downloads come from Hugging Face and happen once per quant.

A **Models folder** entry at the bottom of the picker shows where GGUFs are stored and lets you change it. The default is `~/.local/share/localcode/models`.

## Starting with a model

`localcode --model TAG` starts with an already-downloaded model alias instead of opening the picker. It does not download anything.

## What determines speed

Four things matter, roughly in this order:

1. **Active parameters per token.** On Apple Silicon, memory bandwidth limits decoding speed. A Mixture-of-Experts model uses only a few billion parameters per token. It reads far fewer bytes per token than a dense model with the same total size.
2. **Memory bandwidth.** This varies much more between chip tiers than the number of cores. It directly affects decoding speed.
3. **KV cache size.** TurboQuant compression (`q8_0`-K + `turbo4`-V) keeps the cache small enough for long contexts to remain practical. See [Unified Memory](/localcode/concepts/unified-memory).

4. **Speculative drafting.** Gemma 4, Qwen 3.6 and Muse Glimmer ship with a vendor-trained drafter that downloads with the model; Qwen 3.8 uses a draft head inside its own file. The drafter proposes tokens and the model verifies every one, so the answer is the model's own and decoding is often 1.5 to 2 times as fast. Only drafters trained for the exact model are used.

The model server's GPU kernels ship precompiled, so the first start after an install or update takes seconds rather than 20 to 30. If the shipped kernels do not match your hardware, the server compiles them as before.

localcode does not run a benchmark on your machine and does not publish throughput figures. Real speed depends on your chip, context length, thermal state, and other running tasks.

## Hidden reasoning

Hidden reasoning ("thinking") is off by default for every model, except models such as Flint that cannot turn it off (their thinking is hidden). `/thinking` changes how thinking is displayed when a model produces it.

## Vision

After a vision-capable model loads, a hint says its image projector is available. `/vision` shows the projector's size and downloads it after you confirm. From then on that model can see images: drag or paste one into the prompt, or let the `read` tool open a PNG. The projector is per model.

## Next

- [Unified Memory](/localcode/concepts/unified-memory) - explains the memory budget.
