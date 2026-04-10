# PiKV: KV Cache Management System for Mixture of Experts

- **Authors**: Dong Liu et al.
- **Date**: 2025
- **URL**: https://arxiv.org/abs/2508.06526
- **Code**: https://github.com/NoakLiu/PiKV

## Key Technique

MoE-specific KV cache optimization that jointly optimizes:
1. **Routing**: Token-level saliency scoring
2. **Compression**: Per-expert KV quantization
3. **Eviction**: Inter-expert redundancy detection

For a 7B MoE with 128K context and 16 experts, full KV cache is >24GB.
PiKV treats these three dimensions as a coupled optimization problem.

## Relevance to 64K Goal

**MEDIUM**. Gemma 4 is an MoE model (128 experts, top-8), so MoE-specific
insights are relevant. However, attention in Gemma 4 is NOT per-expert —
K/V are computed once and shared across all expert paths. The MoE routing
only affects the FFN (expert) layers, not the attention mechanism.

So PiKV's per-expert KV management doesn't directly apply to Gemma 4's
architecture. The inter-expert redundancy concept might be useful for
understanding which tokens activate similar expert patterns and could
be safely evicted from the KV cache.

## Implementation Difficulty

**HARD** and **LOW RELEVANCE** for Gemma 4 specifically, since attention
KV is not per-expert.
