# PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference

- **Authors**: Enda Yu, Zhaoning Zhang, Dezun Dong, Yongwei Wu, Xiangke Liao
- **Date**: September 2025
- **URL**: https://arxiv.org/abs/2509.23638

## Key Technique
Three components for prediction-driven cross-layer expert scheduling:
1. **LLaPor** (Learnable Layer-Aware Predictor): Captures layer-specific expert activation patterns
2. **PreSched** (Prefetch-Aware Cross-Layer Scheduling): Generates globally optimal plans balancing prefetching costs and loading overhead
3. **AsyncIO**: Decouples I/O from computation for true overlap

## Results
- 141% higher throughput vs state-of-the-art
- 74.6% lower latency
- Targets commodity hardware with memory/PCIe bottlenecks

## Relevance to 100 tok/s Goal
**HIGH** - The cross-layer scheduling concept is key. Rather than just predicting experts for the next layer, PreScope plans prefetching across MULTIPLE layers ahead. For Gemma 4 with its deep MoE stack, this means we could be loading experts for layer N+2 while computing layer N, then layer N+3 while computing N+1, etc.

The AsyncIO component is directly relevant - on Apple Silicon unified memory, we can use madvise(MADV_WILLNEED) to trigger async prefetching of expert weight pages while the GPU computes on already-loaded experts.

## Implementation Difficulty
**HIGH** - Requires:
- Training per-layer expert predictors
- Cross-layer scheduling optimizer
- Async I/O integration with Metal compute pipeline
- Careful profiling to ensure prefetch completes before compute needs it

## Key Consideration
On Apple Silicon, "I/O" is actually DRAM-to-cache transfer since everything is unified memory. The latency is lower than PCIe, but the bandwidth ceiling is the same ~120 GB/s. The win comes from overlapping cache-line fills with GPU compute rather than serializing them.
