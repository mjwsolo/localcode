# ThunderKittens: Simple, Fast, and Adorable AI Kernels

- **Authors**: Benjamin F. Spector, Simran Arora, et al. (Hazy Research, Stanford)
- **Date**: October 2024
- **URL**: https://arxiv.org/abs/2410.20399
- **Venue**: ICLR 2025

## Key Approach

ThunderKittens (TK) provides a framework for writing performant AI kernels that maps abstractions to three GPU hierarchy levels:

1. **Warp-level**: 16x16 matrix tiles as basic data structures, PyTorch-like parallel compute
2. **Thread-block level**: Template for overlapping asynchronous operations across parallel warps
3. **Grid-level**: Support to hide block launch/tear-down and memory costs

The key insight: by matching abstractions to the GPU memory hierarchy, you get both performance AND readability.

## Results

- Matches FlashAttention-3 on forward passes, outperforms by 10-40% on backward
- Up to 14x over Flash Linear Attention for polynomial variants
- 6.5x over learned feature maps
- 4.7-7.9x over FlashFFTConv for long convolutions
- HipKittens port achieves competitive results on AMD GPUs

## Relation to KG-Optimization Idea

ThunderKittens demonstrates that the right abstraction level is critical:
1. The three-level hierarchy (warp/block/grid) is a natural KG schema for GPU kernels
2. Each abstraction level has its own optimization rules — the KG encodes which optimizations apply at which level
3. TK's DSL could be a **target language** for KG-generated optimizations
4. The portability to AMD (HipKittens) shows that abstract representations transfer across hardware

## What We Can Learn

- Hardware hierarchy should be first-class in the knowledge graph schema
- 16x16 tiles are the fundamental compute unit on modern GPUs
- Matching abstractions to hardware levels is more important than raw cleverness
- A good DSL can make generated kernels inspectable and verifiable
