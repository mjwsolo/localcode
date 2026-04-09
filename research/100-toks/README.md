# 100 tok/s Research

Goal: Push decode speed from 28 tok/s to 100 tok/s on M4 16GB.

## Current State
- 27-28 tok/s decode on M4 16GB MacBook
- Model: Gemma 4 26B IQ3_S (MoE, 3.8B active params)
- Full Metal GPU offload, 2 graph splits
- Memory bandwidth: ~120 GB/s (M4)

## Theoretical Analysis
- Active weights per token: ~1.2 GB (3.8B * ~3 bits)
- Theoretical max: 120 GB/s / 1.2 GB = ~100 tok/s
- Current efficiency: 28% — huge gap to close

## What's Been Tried (from CLAUDE.md)
- Speculative decode: SLOWER for MoE (21.9 vs 27 tok/s)
- Kernel fusion (map0 + matmul): no gain
- Multi-token prediction: Gemma 4 not trained with MTP heads
- Expert pruning (top-4): only 10% gain, quality risk

## Unexplored Approaches
1. **2MB superpages** — reduce TLB misses for scattered expert reads
   - Needs fresh reboot for contiguous physical memory
   - Could eliminate page fault overhead
2. **Expert prefetching** — predict next experts, prefetch weights
3. **Custom Metal kernels** — bypass llama.cpp's generic dispatch
4. **Batch token processing** — process multiple tokens per kernel launch
5. **Weight layout optimization** — reorder GGUF to improve memory access patterns
6. **Shared expert caching** — frequently-used experts stay in fast memory

## Key Files
- `benchmark_toks.py` — decode speed benchmark with profiling
- `metal_profiling/` — Metal GPU profiling data
- `experiments/` — individual experiment results
