---
title: Unified Memory
description: Why RAM, not GPU class, decides which model you can run.
---

On Apple Silicon, the CPU and GPU use the same memory pool. There is no separate VRAM. Model weights, the KV cache, your editor, your browser, and macOS all use this shared memory. This is why localcode bases its model recommendation only on memory size.

## The memory budget

macOS lets the GPU use part of unified memory: about two thirds on Macs up to 36 GB, and about three quarters on larger Macs (107.5 GB on a 128 GB Mac). localcode sizes every model against that limit. Inside it the model needs room for:

- **Weights** - plus any image projector or drafter that loads beside them.
- **KV cache** - It grows as the context gets longer.
- **Activations** - These are temporary, but they still use memory.

The rest of memory stays free for macOS and everything else you have open.

localcode recommends the most capable production-ready model that fits within this budget, and the picker marks every quant as fits, tight or too big. See [Choose a Model](/localcode/start-here/choose-a-model) for the recommended models.

## Keeping the KV cache small

localcode defaults to `q8_0` key and value caches. Its llama.cpp fork also supports **TurboQuant KV cache compression**. With equal key/value head widths, `q8_0` keys plus `turbo4` values use about 2.51× less block storage than `f16` caches, before head padding and adaptive layers. This storage ratio is not a speed measurement.

Compressing the cache can leave more memory for context. LocalCode computes a context limit from the selected model, cache formats and physical RAM; draft models and vision projectors also consume memory. A smaller quant may be required even when the model family is supported.

## Why Mixture-of-Experts models suit mid-range machines

Memory bandwidth limits decode speed on Apple Silicon. This means speed depends on how many bytes the system must read for each token.

An MoE model uses only a few billion of its parameters for each token. It therefore reads much less data per token than a dense model of the same total size. This is the trade-off behind the mid-range recommendations. The actual tokens per second depend on your chip. localcode does not publish throughput figures.

## Memory headroom in practice

- Close memory-heavy apps before a long session, including browsers, IDEs, and Docker.
- A smaller quantisation leaves more memory for a longer context.
- If there is not enough memory to launch, localcode raises `E1010` instead of thrashing.
