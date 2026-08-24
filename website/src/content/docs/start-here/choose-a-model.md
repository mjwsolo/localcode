---
title: Choose a Model
description: How localcode picks a model for your Mac, and how to override it.
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


## Switching models

Type `/model` in the TUI to open the picker and choose another model. `/delete` removes a downloaded model to free disk (it asks first).

If you switch to a model you do not have, localcode downloads it first. You only need to download each model once.



## Next

- [Unified Memory](/localcode/concepts/unified-memory) - explains the memory budget.
- [Models & Performance](/localcode/models-and-performance) - shows the full catalogue.
