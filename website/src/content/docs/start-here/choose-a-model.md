---
title: Choose a Model
description: Which model localcode recommends for your Mac, and how to pick a different one.
---

localcode has no fixed default model. On first launch it checks your Mac's unified memory, marks the recommended model with a star, and lets you choose. Every model in the list runs on the binaries shipped with localcode. The only download is the weights.

## Recommendations by memory

| Unified memory | Recommended | Quant | Weights |
| ---: | --- | --- | ---: |
| 16 GB | Gemma 4 12B | UD-Q4_K_XL | 7.4 GB |
| 24 to 48 GB | Qwen 3.6 35B-A3B | UD-IQ2_M | 10.7 GB |
| 64 GB | Gemma 4 26B-A4B | UD-Q8_K_XL | 28.0 GB |
| 96 GB and up | Qwen 3.6 35B-A3B | UD-Q8_K_XL | 38.5 GB |

The rule: the weights must fit in about 55% of unified memory, leaving room for context and macOS. Among the models that fit, localcode recommends the most capable one.

The full list of models, with sizes and minimum memory, is in [Models & Performance](/localcode/models-and-performance).

## Why the mid-range picks are Mixture-of-Experts

A Mixture-of-Experts model only uses a few billion parameters for each token, so it generates text about as fast as a much smaller model while keeping the quality of a larger one. That is what makes a 35B model practical on a laptop.

## Switching models

Inside the app:

```text
/model              # list models
/model qwen         # switch
/delete             # remove a downloaded model to free disk space
```

From the command line:

```sh
localcode --model <tag>
```

Switching to a model you have not downloaded starts the download.

## Picking by hand

The picker also lists other quantisations of each model. A smaller quant uses less memory and leaves room for a longer context. A larger quant gives better answers. You can pick a model that is heavier than the recommendation; the fit check is advice, not a limit.

DiffusionGemma is a research model that is never recommended automatically. It runs through a separate bundled binary instead of the normal server.

## Next

- [Unified Memory](/localcode/concepts/unified-memory)
- [Models & Performance](/localcode/models-and-performance)
