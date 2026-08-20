---
title: Unified Memory
description: Why RAM, not GPU class, decides which model you can run.
---

On Apple Silicon, the CPU and GPU use the same memory pool. There is no separate VRAM. Model weights, the KV cache, your editor, your browser, and macOS all use this shared memory. This is why localcode bases its model recommendation only on memory size.

## The memory budget

localcode uses about **55% of unified memory** for model weights. The remaining memory must cover:

- **KV cache** — It grows as the context gets longer and can use a lot of laptop memory.
- **Activations** — These are temporary, but they still use memory.
- **macOS and everything else you have open.**

localcode recommends the most capable production-ready model whose weights fit within this budget. See [Choose a Model](/localcode/start-here/choose-a-model) for the recommended models.

## Why the KV cache uses so much memory

localcode uses a llama.cpp fork with **TurboQuant KV cache compression**. It uses asymmetric `q8_0`-K and `turbo4`-V quantisation. According to the fork, this combination is about 3.8× smaller than `f16`. This figure describes the quantisation method, not your machine's performance.

Compressing the cache leaves more memory for a longer context. This is why you can configure the K/V cache types with `kv_cache_type_k` and `kv_cache_type_v`.

## Why Mixture-of-Experts models suit mid-range machines

Memory bandwidth limits decode speed on Apple Silicon. This means speed depends on how many bytes the system must read for each token.

An MoE model uses only a few billion of its parameters for each token. It therefore reads much less data per token than a dense model of the same total size. This is the trade-off behind the mid-range recommendations. The actual tokens per second depend on your chip. localcode does not publish throughput figures.

## Memory headroom in practice

- Close memory-heavy apps before a long session, including browsers, IDEs, and Docker.
- A smaller quantisation leaves more memory for a longer context.
- If there is not enough memory to launch, localcode raises `E1010` instead of thrashing.
