# SqueezeAttention: 2D Management of KV-Cache via Layer-wise Optimal Budget

- **Authors**: Various (ICLR 2025)
- **Date**: April 2024 (ICLR 2025)
- **URL**: https://arxiv.org/abs/2404.04793
- **Code**: https://github.com/hetailang/SqueezeAttention

## Key Technique

Optimizes KV cache from two dimensions simultaneously:
1. **Sequence-wise**: Which tokens to keep (like H2O/ScissorHands)
2. **Layer-wise**: How much budget each layer gets

Measures layer importance via cosine similarity of input differences before
and after self-attention. Groups layers by importance and adjusts budgets.

## Compression Ratio

30-70% memory reduction with up to 2.2x throughput improvement.

## Relevance to 64K Goal

**HIGH — key architectural insight for Gemma 4**. With 30 layers total:
- 26 sliding window layers (fixed small cache — already optimized)
- 4 global layers — these are the bottleneck

SqueezeAttention's layer-wise budgeting could reveal that not all 4 global
layers need the same amount of KV cache. If 2 of the 4 global layers are
less critical, we could use smaller windows or more aggressive quantization
for those layers.

At 64K, reducing even one global layer's effective context by half saves
significant memory since global layers use 512 head_dim (2x the sliding layers).

## Implementation Difficulty

**MEDIUM**. The profiling step (cosine similarity measurement) is
straightforward. The challenge is implementing per-layer KV budgets in
llama.cpp, which currently allocates uniformly.
