---
title: Choose a Model
description: How localcode picks a model for your Mac, and how to override it.
---

localcode has **no fixed default model**. On launch it reads your Mac's unified
memory and recommends the most capable production-ready model whose weights fit
the memory budget.

## The rule

Model weights must fit in about **55% of unified memory**. The remaining
headroom goes to the KV cache, activations and macOS. Among the models that
fit, localcode picks the most capable one; experimental architectures are never
auto-recommended, though you can still choose them by hand.

## What that means per Mac

| Unified memory | Recommended | Quant | Weights |
| ---: | --- | --- | ---: |
| 16 GB | Gemma 4 12B | UD-Q4_K_XL | 7.37 GB |
| 24–48 GB | Qwen 3.6 35B-A3B | Q2 | 10.7 GB |
| 64 GB | Gemma 4 26B-A4B | Q8 | 28.0 GB |
| 96 GB+ | Qwen 3.6 35B-A3B | UD-Q8_K_XL | 38.5 GB |

These rows are what `localcode.models_catalog.recommend()` returns for each
memory size — they are not hand-tuned marketing picks. You can reproduce them:

```sh
python -c "from localcode import models_catalog as m; c = m.recommend(32); print(c.name, c.size_gb)"
```

## Why the mid-range pick is a Mixture-of-Experts model

The 24–96 GB recommendations are MoE models: only a few billion parameters are
active per token even though the model is much larger. Decode speed tracks the
*active* parameters, so an MoE model gives you a bigger model's capability at a
smaller model's per-token cost. That is what makes the larger picks practical
on a laptop at all.

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

Switching to a model you don't have downloads it first. Each model is a
one-time download; see [Network Boundary](/localcode/concepts/network-boundary).

## Experimental models

Some catalogued models are pickable but never auto-recommended, because their
architecture needs a runner other than the bundled server — a diffusion model,
for example, or an architecture the bundled fork doesn't implement. localcode
builds a dedicated server for those on first use. Treat them as experiments,
not as the paved path.

## Choosing by hand

The picker also lets you browse other quantisations of a supported model
family. A smaller quant frees memory for a longer context; a larger quant
costs memory and buys quality. If a model's weights exceed your memory budget
localcode will still let you pick it — the fit check is advice, not a lock.

## Next

- [Unified Memory](/localcode/concepts/unified-memory) — why the budget is what it is.
- [Models & Performance](/localcode/models-and-performance) — the full catalogue.
