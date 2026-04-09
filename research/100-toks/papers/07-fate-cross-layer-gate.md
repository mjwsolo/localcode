# Fate: Fast Edge Inference of MoE Models via Cross-Layer Gate

- **Authors**: Zhiyuan Fang, Zicong Hong, Yuegui Huang, Yufeng Lyu, Wuhui Chen, Yue Yu, Fan Yu, Zibin Zheng
- **Date**: February 2025 (revised May 2025)
- **URL**: https://arxiv.org/abs/2502.12224

## Key Technique
Uses gate inputs from adjacent layers to predict expert selection for the next layer. Cross-layer gate predictions achieve high accuracy without requiring separate predictor models - just reuses the router's own input patterns across layers.

## Relevance to 100 tok/s Goal
**HIGH** - Simpler than training separate predictors (paper 02). If Gemma 4's layer N gate inputs can predict layer N+1's expert selection with >90% accuracy, this gives us prefetching capability with near-zero computational overhead.

For our mmap-based setup, accurate prediction means we can issue madvise(MADV_WILLNEED) for the next layer's expert weight pages while the current layer computes. This could overlap ~50% of memory reads with computation.

## Implementation Difficulty
**LOW-MEDIUM** - No training needed. Just:
1. Profile Gemma 4's cross-layer expert correlation on representative inputs
2. If correlation is high enough (>85%), implement simple cross-layer prediction
3. Add prefetch hints (madvise or manual cache warming) between layers

This is the lowest-hanging fruit among all prediction approaches.

## Risk
Gemma 4 has 128 experts (much more than Mixtral's 8). Cross-layer correlation might be weaker with more experts. Needs profiling before committing.
