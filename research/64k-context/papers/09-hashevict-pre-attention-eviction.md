# HashEvict: A Pre-Attention KV Cache Eviction Strategy using Locality-Sensitive Hashing

- **Authors**: Minghui Liu, Tahseen Rabbani, Tony O'Halloran, et al.
- **Date**: December 2024 (revised June 2025)
- **URL**: https://arxiv.org/abs/2412.16187

## Key Technique

Uses locality-sensitive hashing (LSH) to identify which cached tokens are
dissimilar to the current query BEFORE computing attention. Eviction happens
pre-attention, avoiding the cost of attending to tokens that will be pruned.

Maintains a lightweight binary hash structure in GPU memory. Computes Hamming
distance between binarized Gaussian projections of current and cached tokens.

## Compression Ratio

30-70% KV cache compression while preserving quality.

## Relevance to 64K Goal

**MEDIUM**. Could serve as the eviction policy for our global attention layers.
At 64K context, the 4 global layers need full-range attention. HashEvict could
dynamically evict low-importance tokens from global layers, keeping effective
cache size smaller.

Combines well with quantization: quantize what you keep, evict what you don't need.

Pre-attention eviction is particularly valuable because it saves both memory
AND compute — you don't waste attention cycles on tokens about to be evicted.

## Implementation Difficulty

**MEDIUM-HARD**. Requires:
1. LSH projection matrix per layer (small, precomputed)
2. Binary hash maintenance during generation
3. Custom eviction logic in llama.cpp attention loop
4. Tuning eviction threshold per model

The pre-attention aspect adds latency to the eviction decision, though the
hash comparison is fast (Hamming distance on binary vectors).
