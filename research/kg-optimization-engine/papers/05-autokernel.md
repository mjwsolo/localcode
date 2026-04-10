# AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search

- **Authors**: Jaber Jaber, Osama Jaber (RightNow AI)
- **Date**: March 2026
- **URL**: https://arxiv.org/abs/2603.21331
- **Venue**: arXiv

## Key Approach

AutoKernel is pragmatic and iterative: an LLM agent modifies a single kernel file, a fixed benchmark validates correctness and measures performance, and outcomes determine whether changes are retained or discarded. Uses Amdahl's law to prioritize — focuses agent effort on kernels consuming the most GPU time.

Three phases:
1. **Model profiling** identifies computational bottlenecks
2. **Agent optimization loop** iteratively refines Triton or CUDA implementations through hundreds of experiments
3. **Verification** through five stages: smoke tests, shape sweeps, numerical stability, determinism, edge cases

Key differentiator: dual Triton/CUDA C++ backend support with model-level profiling.

## Results (H100)

- RMSNorm: 5.29x over eager, 2.83x over torch.compile
- Softmax: 2.82x over eager, 3.44x over torch.compile
- Cross-entropy: 2.21x over eager, 2.94x over torch.compile
- Won first place on vectorsum_v2 B200 leaderboard
- Triton FP4 matmul: 1.63-2.15x over CUTLASS

## Relation to KG-Optimization Idea

AutoKernel's Amdahl's-law-guided prioritization is exactly what a knowledge graph enables at scale:
1. The KG encodes the full inference pipeline, making bottleneck identification automatic
2. Profiling data feeds back into the graph, creating a self-improving system
3. Verified optimizations become reusable knowledge nodes
4. The five-stage verification pipeline is a template for our validation system

## What We Can Learn

- Amdahl's law prioritization prevents wasted effort on low-impact kernels
- Dual backend support (Triton + CUDA) widens the optimization space significantly
- Rigorous five-stage verification is essential for production use
- Simple iterative approach with good tooling outperforms complex architectures
