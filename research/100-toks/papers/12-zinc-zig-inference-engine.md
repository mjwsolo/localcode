# ZINC: Zig INferenCe Engine

- **Author**: Stepan Zolotukhin
- **Date**: 2025-2026
- **URL**: https://github.com/zolotukhin/zinc

## Key Technique
From-scratch LLM inference engine in Zig with hand-tuned Metal shaders:
- Native MSL kernels with simdgroup reductions
- Zero-copy model loading
- Metal pipeline tuning
- No framework overhead (no Python, no CUDA, no ROCm)

## Performance
- ~34.7 tok/s on Apple M4 Max with a 35B dense model
- ~33.5 tok/s on RDNA4 with same model

## Relevance to 100 tok/s Goal
**MEDIUM-HIGH** - ZINC achieves ~35 tok/s on M4 Max (400 GB/s bandwidth) with a 35B model. That's also about 28% bandwidth utilization -- similar to our llama.cpp ratio. This suggests the 28% utilization is NOT just a llama.cpp problem but a fundamental Metal/GPU overhead issue.

Key implications:
1. Even a from-scratch engine with hand-tuned Metal shaders hits the same ~28% wall
2. The bottleneck is likely in Metal's dispatch overhead, command buffer encoding, or GPU memory controller behavior
3. Going faster may require fundamentally different compute patterns (batching multiple tokens, fusing more operations into single kernel dispatches)

However, M4 Max has ~3.3x our bandwidth (400 vs 120 GB/s). If ZINC achieves 35 tok/s on 400 GB/s, that's ~10.5 tok/s equivalent at 120 GB/s -- worse than our 28 tok/s. So our llama.cpp setup may actually be more efficient per-bandwidth than ZINC on MoE (because our model is smaller per active weight).

## Implementation Difficulty
**VERY HIGH** - Rewriting the inference engine from scratch. Not practical. But studying ZINC's Metal shader patterns could reveal optimization opportunities.
