# SWAA: Sliding Window Attention Adaptation for Efficient Long-Context

- **Authors**: Yijiong Yu, Jiale Liu, Qingyun Wu, Huazheng Wang, Ji Pei
- **Date**: December 2025 (revised March 2026)
- **URL**: https://arxiv.org/abs/2512.10411

## Key Technique

Adapts full-attention models to use sliding window attention without costly
pretraining. Four strategies:
1. Full Attention Decode during inference
2. Interleaving full and sliding window attention layers
3. Preserving "sink" tokens
4. Lightweight fine-tuning

Plug-and-play toolkit — no full retraining needed.

## Performance

- 30-100% faster inference for long contexts
- Acceptable quality retention
- Reduces quadratic complexity toward linear

## Relevance to 64K Goal

**LOW**. Gemma 4 already has a hybrid sliding/global architecture by design
(26 sliding + 4 global layers). SWAA solves the problem of adapting
full-attention models to SWA, which isn't our problem. Our model was
trained with this pattern natively.

However, the interleaving strategy validates that Gemma 4's approach
(mostly sliding with a few global layers) is sound.

## Implementation Difficulty

N/A — not needed for Gemma 4 since it already uses this architecture.
