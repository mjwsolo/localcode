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

These come straight from `localcode.models_catalog.recommend()`, not a hand-picked list. Reproduce any of them:

```sh
python -c "from localcode import models_catalog as m; c = m.recommend(32); print(c.name, c.size_gb)"
```

## Why the mid-range picks are Mixture-of-Experts

The 24-96 GB recommendations use MoE models. Only a few billion parameters are active per token, even though the full model is much larger, and decode speed depends on those active parameters. You get the quality of a large model at the per-token cost of a small one, which is what makes these models practical on a laptop.

## Switching models

Inside the TUI:

```text
/model              # list what's available
/model qwen         # switch
/delete             # remove a downloaded model to free disk (asks first)
```

From the command line:

```sh
localcode --model <tag>
```

If you switch to a model you do not have, localcode downloads it first. You only need to download each model once. See [Network Boundary](/localcode/concepts/network-boundary).

## Experimental models

You can choose some models from the catalogue even though localcode never recommends them automatically. Their architecture needs a different runner from the bundled server. Examples include diffusion models and architectures that the bundled fork does not support. localcode builds a dedicated server the first time you use one. These models are experimental and are not the standard choice.

## Choosing by hand

The picker also shows other quantisations for supported model families. A smaller quant uses less memory and allows a longer context. A larger quant uses more memory but gives better quality. You can still choose a model whose weights exceed your memory budget. The fit check is advice, not a lock.

## Next

- [Unified Memory](/localcode/concepts/unified-memory) - explains the memory budget.
- [Models & Performance](/localcode/models-and-performance) - shows the full catalogue.
