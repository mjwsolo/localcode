---
title: Unified Memory
description: Why the amount of RAM, not the GPU, decides which model you can run.
---

On Apple Silicon the CPU and GPU share one pool of memory. There is no separate video memory. Model weights, the model's working memory, your editor, your browser and macOS all draw from the same pool. That is why localcode recommends a model by memory size alone.

## The budget

localcode recommends models whose weights use about **55% of unified memory** or less. The rest goes to:

- **The KV cache.** The model's memory of the conversation. It grows with context length.
- **Activations.** Temporary working memory during generation.
- **macOS and everything else you have open.**

You can change how the KV cache is stored with `kv_cache_type_k` and `kv_cache_type_v` in [Configuration](/localcode/reference/configuration). A more compressed cache leaves room for a longer context.

## Why Mixture-of-Experts models suit mid-range Macs

Generation speed on Apple Silicon is limited by how many bytes the chip reads per token. A Mixture-of-Experts model reads only its active experts, a few billion parameters, so it generates about as fast as a small model while keeping the quality of a large one. See [Choose a Model](/localcode/start-here/choose-a-model).

## Practical tips

- Close memory-heavy apps such as browsers, IDEs and Docker before a long session.
- A smaller quantisation leaves room for a longer context.
- If there is not enough memory to launch the model, localcode reports `E1010` rather than thrashing.
