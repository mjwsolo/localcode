# KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models

- **Authors**: Tsinghua University MADSys Group
- **Date**: 2025 (SOSP '25)
- **URL**: https://dl.acm.org/doi/10.1145/3731569.3764843

## Key Technique
CPU/GPU hybrid inference with three innovations:
1. **AMX-optimized kernels**: Tiling-aware memory layout + cache-optimized AMX kernels for CPU expert computation
2. **Asynchronous CPU-GPU scheduling**: Minimizes synchronization overhead between CPU expert compute and GPU attention compute
3. **Expert Deferral**: Strategically delays some expert computations to maximize CPU-GPU overlap, increasing CPU utilization from <75% to ~100% with <0.5% accuracy drop

## Results
- 4.62-19.74x prefilling speedups
- 1.25-4.09x decoding speedups
- Runs DeepSeek-R1 671B on single 24GB GPU + 382GB DRAM
- SGLang + KTransformers: 220+ tok/s total throughput on trillion-parameter models

## Relevance to 100 tok/s Goal
**MEDIUM** - KTransformers is designed for discrete GPU + CPU DRAM, not Apple Silicon unified memory. However, two ideas transfer:

1. **Expert Deferral**: Skip non-critical experts during decode and approximate their output. If some of the 8 active experts contribute minimally, deferring 1-2 could reduce active weight reads by 12-25%.

2. **Compute/memory overlap**: The principle of running CPU expert compute in parallel with GPU attention is analogous to what we need - overlapping expert weight fetches with attention computation on the same unified memory bus.

## Implementation Difficulty
**MEDIUM-HIGH** - Expert deferral requires profiling expert contribution significance and implementing a skip/approximate mechanism. The AMX-specific optimizations don't apply to Apple Silicon (we'd need Metal equivalents).

## Key Limitation
Apple Silicon doesn't have AMX (it has AMX-equivalent in the Neural Engine, but that's not exposed for arbitrary compute). The CPU/GPU split doesn't apply when both share unified memory. But the expert deferral concept is architecture-agnostic.
