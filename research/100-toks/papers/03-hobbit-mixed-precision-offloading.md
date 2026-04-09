# HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference

- **Authors**: Peng Tang, Jiacheng Liu, Xiaofeng Hou, Yifei Pu, Jing Wang, Pheng-Ann Heng, Chao Li, Minyi Guo
- **Date**: November 2024
- **URL**: https://arxiv.org/abs/2411.01433

## Key Technique
Three-level hierarchy of optimizations:
1. **Token-level**: Dynamic expert loading - replace cache-miss experts with low-precision versions on the fly
2. **Layer-level**: Adaptive expert prefetching across model layers
3. **Sequence-level**: Multidimensional expert caching policy

Core insight: dynamically replacing less critical cache-miss experts with low-precision versions reduces expert-loading latency while preserving accuracy.

## Results
- Up to 9.93x speedup in decoding vs state-of-the-art MoE offloading systems
- Built on llama.cpp framework
- Tested on edge devices with memory constraints

## Relevance to 100 tok/s Goal
**HIGH** - The mixed-precision approach is directly applicable. For Gemma 4 at IQ3_S, we could maintain two copies of expert weights: the full IQ3_S for "hot" experts and an even more aggressive IQ2 or IQ1 for "cold" experts. During decode, if a predicted expert misses cache, fall back to the ultra-low-precision version instead of waiting for the full read. This trades tiny quality loss for major latency reduction.

The llama.cpp base means integration path is clear for our custom fork.

## Implementation Difficulty
**HIGH** - Requires:
- Dual quantization of all expert weights (IQ3_S + IQ1_S)
- Runtime decision logic for precision selection
- Modified Metal kernels for both precision levels
- Increased storage (but expert weights are small per-expert)

## Key Insight for Our Setup
Since we're already at IQ3_S (aggressive quant), the "low precision fallback" would need to be IQ2_S or even binary experts. Quality impact needs careful measurement. But even a simple hot/cold expert cache (keep 16-32 most frequent experts in fast-access pattern) could help.
