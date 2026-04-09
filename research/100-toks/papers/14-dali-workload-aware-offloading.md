# DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs

- **Authors**: (Multiple authors)
- **Date**: February 2026
- **URL**: https://arxiv.org/abs/2602.03495

## Key Technique
Three-pronged optimization:
1. **Greedy Assignment**: Dynamic CPU/GPU expert assignment modeled as 0-1 integer optimization, solved greedily at runtime
2. **Residual-Based Prefetching**: Uses inter-layer residual information (skip connections) to predict which experts will be heavy-load in the next layer
3. **Workload-Aware Cache Replacement**: Updates expert cache based on workload history of previous tokens, exploiting temporal correlation

## Key Insight: Residual-Based Prediction
The residual stream (skip connection output) contains information about which experts will be needed next. This is similar to the cross-layer gate prediction (paper 07) but uses the residual directly rather than the gate input. This could be even simpler to implement since residuals are always available.

## Results
- Compares favorably against llama.cpp, KTransformers, MoE-Lightning, HybriMoE
- Achieves speedups in both prefill and decode phases

## Relevance to 100 tok/s Goal
**HIGH** - The residual-based prediction is particularly interesting because:
1. Gemma 4 uses pre-norm + residual architecture
2. The residual stream is computed BEFORE the MoE layer
3. No additional computation needed -- just use the residual to predict next-layer experts
4. Could enable zero-overhead expert prefetching

The workload-aware cache replacement also matters for our SLC optimization strategy.

## Implementation Difficulty
**MEDIUM** - Residual-based prediction is cheap (just a small linear layer on the residual). Integration with madvise prefetching is straightforward. The main work is profiling prediction accuracy for Gemma 4's specific architecture.
