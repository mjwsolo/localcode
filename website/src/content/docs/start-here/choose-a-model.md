---
title: Models
description: How localcode picks a model for your Mac, how to override it, and what determines speed.
---

localcode has **no fixed default model**. When it starts, it checks your Mac's unified memory. It then recommends the most capable production-ready model whose weights fit in the available memory.

## The rule

Model weights must use about **55% of unified memory** or less. The rest is for the KV cache, activations, and macOS. localcode recommends the most capable model that fits. It never recommends experimental architectures automatically, but you can choose them yourself.

## Recommendations for each Mac

| Unified memory | Recommended | Quant | Weights |
| ---: | --- | --- | ---: |
| 16 GB | Gemma 4 12B | UD-Q4_K_XL | 7.37 GB |
| 24–48 GB | Qwen 3.6 35B-A3B | Q2 | 10.7 GB |
| 64 GB | Gemma 4 26B-A4B | Q8 | 28.0 GB |
| 96 GB+ | Qwen 3.6 35B-A3B | UD-Q8_K_XL | 38.5 GB |


## What determines speed

Three things matter, roughly in this order:

1. **Active parameters per token.** On Apple Silicon, memory bandwidth limits decoding speed. A Mixture-of-Experts model uses only a few billion parameters per token. It reads far fewer bytes per token than a dense model with the same total size.
2. **Memory bandwidth.** This varies much more between chip tiers than the number of cores. It directly affects decoding speed.
3. **KV cache size.** TurboQuant compression (`q8_0`-K + `turbo4`-V) keeps the cache small enough for long contexts to remain practical. See [Unified Memory](/localcode/concepts/unified-memory).

## The tok/s numbers in the model picker

The model picker shows an estimated decoding speed next to each quantisation. This number is **calculated, not measured**. localcode does not run a benchmark on your machine. There is no benchmark command or benchmark screen.

The estimate uses an analytic model. It divides the bytes read per token by an assumed share of your chip's rated memory bandwidth. It then adds a fixed compute time for each token. The bytes per token come from the quant's size and its active-parameter fraction. This means MoE models count only their active experts. The model is calibrated using a small number of maintainer measurements from one machine.

Use the estimate to compare options. It can show that one quant will be slower than another on your hardware. Do not treat it as a prediction of your actual throughput. Real speed also depends on context length, thermal state, and other running tasks.

## Switching models

Type `/model` in the TUI to open the picker and choose another model. `/delete` removes a downloaded model to free disk (it asks first).

If you switch to a model you do not have, localcode downloads it first. You only need to download each model once.



## Next

- [Unified Memory](/localcode/concepts/unified-memory) - explains the memory budget.
