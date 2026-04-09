# A Survey on Inference Optimization Techniques for Mixture of Experts Models

- **Authors**: Jiacheng Liu, Peng Tang, Wenfeng Wang, Yuhang Ren, Xiaofeng Hou, Pheng-Ann Heng, Minyi Guo, Chao Li
- **Date**: December 2024 (revised January 2025)
- **URL**: https://arxiv.org/abs/2412.14219

## Key Technique
Comprehensive taxonomy of MoE inference optimizations across three levels:
1. **Model-level**: Expert pruning, quantization, knowledge distillation, dynamic routing, expert merging
2. **System-level**: Distributed computing, load balancing, efficient scheduling
3. **Hardware-level**: Hardware-specific co-design for throughput and energy efficiency

## Key Finding for Our Setup
MoE FFNs sustain low SM utilization (28-34%) and high DRAM pressure (>80% at small batches). Dense FFN arithmetic intensity reaches 15.74 FLOP/byte, while MoE stays at 8 FLOP/byte because routing distributes batches across experts creating tiny per-expert batches. This is the fundamental reason our bandwidth utilization is only 28%.

## Relevance to 100 tok/s Goal
**HIGH** - Explains exactly why we're at 28% bandwidth utilization. The survey maps out all known optimization approaches. Our situation (single GPU, model fits in memory, decode-bound) points toward: expert caching, weight layout optimization, and reducing per-token memory reads.

## Implementation Difficulty
N/A (survey paper) - but identifies which techniques are most promising for our constraints.
