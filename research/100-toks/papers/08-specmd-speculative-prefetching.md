# SpecMD: A Comprehensive Study on Speculative Expert Prefetching

- **Authors**: (Multiple authors)
- **Date**: February 2026
- **URL**: https://arxiv.org/abs/2602.03921

## Key Technique
Standardized benchmarking framework for MoE caching strategies. Key finding: dynamic score-based prefetching outperforms top-k approaches. Despite lower overall prediction accuracy, score-based prefetching yields higher hit rates because it makes better use of available bandwidth.

## Key Finding
Score-based prefetching > top-k prefetching. The router's softmax scores already contain rich information about expert likelihood. Instead of predicting binary "which experts", use the score distribution to prefetch experts in order of decreasing probability, filling available bandwidth.

## Relevance to 100 tok/s Goal
**HIGH** - This is directly applicable. During decode, while computing layer N's selected experts:
1. Look at layer N+1's router scores (from cross-layer prediction)
2. Sort experts by predicted score
3. Prefetch in score order, filling available memory bandwidth
4. Even if prediction misses some experts, the highest-score ones are most likely correct

This probabilistic approach is more robust than binary prediction for our 128-expert setup.

## Implementation Difficulty
**MEDIUM** - Requires:
- Router score extraction during inference
- Score-ordered prefetch queue
- Integration with madvise/prefetch hints
- Profiling to determine optimal prefetch depth (how many experts to speculatively load)
