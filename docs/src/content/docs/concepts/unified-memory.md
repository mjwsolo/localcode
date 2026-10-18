---
title: Unified Memory
description: Why RAM, not GPU class, decides which model you can run.
---

On Apple Silicon, the CPU and GPU use the same memory pool. There is no separate VRAM. Model weights, the KV cache, your editor, your browser, and macOS all use this shared memory. This is why localcode bases its model recommendation only on memory size.

## The memory budget

macOS lets the GPU keep only part of unified memory in use at once: about two thirds on smaller Macs and more on larger ones (`iogpu.wired_limit_mb` overrides it when set). localcode reads this limit from your Mac and never raises it. Everything a fully offloaded model needs must sit inside it:

- **Model weights**, plus the speculative drafter when one loads.
- **KV cache** - It grows as the context gets longer. Hybrid models (Qwen 3.6 and 3.8) keep it only for every fourth layer, and Gemma 4 keeps most layers to a 1024-token window, so their caches are small.
- **Working buffers** - About 1 GB.

macOS and everything else you have open use the rest. localcode recommends the most capable production-ready model whose weights fit within this limit beside a 32K-token working context. See [Choose a Model](/localcode/start-here/choose-a-model) for the recommended models.

## Keeping the KV cache small

localcode defaults to `q8_0` key and value caches. Its llama.cpp fork also supports **TurboQuant KV cache compression**. With equal key/value head widths, `q8_0` keys plus `turbo4` values use about 2.51× less block storage than `f16` caches, before head padding and adaptive layers. This storage ratio is not a speed measurement.

Compressing the cache can leave more memory for context. LocalCode computes a context limit from the selected model, cache formats, physical RAM and the GPU limit; draft models and vision projectors also consume memory. The vision projector loads only once an image is sent, and a drafter that does not fit is left out. A smaller quant may be required even when the model family is supported.

## Why Mixture-of-Experts models suit mid-range machines

Memory bandwidth limits decode speed on Apple Silicon. This means speed depends on how many bytes the system must read for each token.

An MoE model uses only a few billion of its parameters for each token. It therefore reads much less data per token than a dense model of the same total size. This is the trade-off behind the mid-range recommendations. The actual tokens per second depend on your chip. localcode does not publish throughput figures.

## Memory headroom in practice

- Close memory-heavy apps before a long session, including browsers, IDEs, and Docker.
- A smaller quantisation leaves more memory for a longer context.
- If there is not enough memory to launch, localcode raises `E1010` instead of thrashing.
