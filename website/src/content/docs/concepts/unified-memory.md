---
title: Unified Memory
description: Why RAM, not GPU class, decides which model you can run.
---

On Apple Silicon, the CPU and GPU share one pool of memory. There is no
separate VRAM to fill — the model weights, the KV cache, your editor, your
browser and macOS all draw from the same pool. That is why localcode's model
recommendation is a function of memory size and nothing else.

## The budget

localcode allocates roughly **55% of unified memory** to model weights. The
rest has to cover:

- **KV cache** — grows with context length, and is the thing that quietly
  eats a laptop.
- **Activations** — transient, but real.
- **macOS and everything else you have open.**

Among the production-ready models whose weights fit that budget, localcode
recommends the most capable. See
[Choose a Model](/localcode/start-here/choose-a-model) for the resulting picks.

## Why the KV cache matters so much

localcode runs a llama.cpp fork with **TurboQuant KV cache compression**:
asymmetric `q8_0`-K plus `turbo4`-V quantisation, about 3.8× smaller than
`f16`. Compressing the cache is what buys back enough headroom to run a useful
context on a 16 GB machine, and it is why the K/V cache types are exposed as
configuration (`kv_cache_type_k`, `kv_cache_type_v`).

## Why Mixture-of-Experts models are the mid-range pick

Decode speed on Apple Silicon is bounded by memory bandwidth — how many bytes
have to be read per token. An MoE model activates only a few billion of its
parameters per token, so it reads far fewer bytes than a dense model of the
same total size. You get a larger model's capability at a smaller model's
per-token bandwidth cost, which is exactly the trade a laptop wants.

## Headroom, in practice

- Quit the memory-heavy apps before a long session — browsers, IDEs, Docker.
- A smaller quantisation frees memory for a longer context.
- If a launch fails on memory, localcode raises `E1010` rather than thrashing.
