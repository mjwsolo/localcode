# SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization

- **Authors**: (Multiple, 2025)
- **Date**: August 2025
- **URL**: https://arxiv.org/abs/2508.20258
- **Venue**: arXiv

## Key Approach

SwizzlePerf adds rich profiling context from a suite of GPU profilers to the LLM, directly reflecting cache-locality improvements. Instead of giving the LLM just source code and runtime, it provides detailed hardware performance counters, cache hit rates, memory access patterns, and occupancy metrics.

The key insight: LLMs can reason about optimization when given the right hardware feedback, but raw code alone doesn't provide enough signal for hardware-aware decisions.

## Relation to KG-Optimization Idea

SwizzlePerf's profiling data is exactly what should populate our knowledge graph:
1. Cache-locality metrics, occupancy data, memory bandwidth utilization become node properties
2. The profiler suite provides the "sensor data" for our optimization graph
3. Profiling feedback loops create training data for the KG agent
4. Hardware performance counters are the ground truth for optimization decisions

## What We Can Learn

- Raw code is insufficient context for hardware optimization — profiling data is essential
- A suite of profilers (not just one) provides multi-dimensional optimization signals
- Cache-locality is a particularly informative signal for kernel optimization
- The profiling-feedback-optimization loop should be automated end-to-end
