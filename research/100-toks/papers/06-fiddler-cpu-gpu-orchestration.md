# Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models

- **Authors**: EfeSlab (University of Michigan)
- **Date**: February 2024 (ICLR 2025)
- **URL**: https://arxiv.org/abs/2402.07033

## Key Technique
Instead of moving expert weights to GPU, move activations to CPU and compute experts on CPU. Key insight: for small batch sizes, activation tensors are much smaller than expert weight tensors, so it's faster to move activations than weights.

Uses a latency model to dynamically decide: run expert on GPU (if weights cached) or send activations to CPU (if weights not in GPU memory).

## Results
- Runs uncompressed Mixtral-8x7B (90GB+) on single 24GB GPU
- 3+ tok/s generation
- 8.2x and 10.1x speedup vs baselines for single-batch inference

## Relevance to 100 tok/s Goal
**LOW-MEDIUM** - The activation-shipping-to-CPU approach doesn't directly apply to Apple Silicon unified memory (no PCIe bottleneck to avoid). However, the core insight about batch-size-dependent optimal execution strategy is relevant:

For our single-batch decode, each expert processes a tiny activation (just 1 token). The matmul is a matrix-vector product. This means:
- The operation is entirely memory-bandwidth-bound
- Compute intensity is minimal
- Any technique that reduces the bytes read per expert per token directly translates to speed

## Implementation Difficulty
N/A for direct use, but the analytical framework for understanding when compute vs memory dominates is useful for our optimization decisions.
