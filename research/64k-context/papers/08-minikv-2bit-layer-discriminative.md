# MiniKV: Pushing the Limits of 2-Bit KV Cache

- **Authors**: Akshat Sharma, Hangliang Ding, Jianping Li, Neel Dani, Minjia Zhang
- **Date**: November 2024 (ACL 2025 Findings)
- **URL**: https://arxiv.org/abs/2411.18077

## Key Technique

Two-phase approach:
1. **Prefill phase**: Pyramid KV with rectified token selection — different
   layers get different KV budgets based on importance
2. **Important tokens**: Sub-channel Key quantization + per-token Value
   quantization at 2-bit

Layer-discriminative: not all layers need the same KV cache size. Some layers
are more sensitive and get larger budgets; others can be aggressively compressed.

## Compression Ratio

2-bit KV cache with layer-wise budget allocation.
Effective compression depends on the layer budget distribution.

## Relevance to 64K Goal

**MEDIUM-HIGH**. The layer-discriminative insight is valuable for Gemma 4:
- 26 sliding layers already have small fixed-size caches
- 4 global layers could have different budgets based on importance
- Combining 2-bit quantization with unequal layer budgets could yield
  better quality than uniform 2-bit across all layers

This is complementary to turbo2/turbo3 — apply more aggressive compression
to less important layers.

## Implementation Difficulty

**MEDIUM-HARD**. Requires:
1. Profiling per-layer importance for Gemma 4
2. Per-layer KV cache size configuration in llama.cpp (not currently supported)
3. Different quantization levels per layer
