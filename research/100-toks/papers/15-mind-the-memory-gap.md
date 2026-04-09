# Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference

- **Authors**: IBM Research
- **Date**: March 2025
- **URL**: https://arxiv.org/abs/2503.08311

## Key Findings
- Large-batch inference remains memory-bound even with high batch sizes
- GPU compute capabilities are severely underutilized due to DRAM bandwidth saturation
- The "memory gap" between compute capability and memory bandwidth is the primary bottleneck

## Relevance to 100 tok/s Goal
**MEDIUM** - Confirms that the memory bandwidth wall is real and applies even to high-end GPUs. For our M4 at 120 GB/s with single-batch decode, we're in the worst-case scenario (batch=1, pure GEMV, minimum arithmetic intensity).

The paper suggests that the only paths beyond the bandwidth wall are:
1. **Reduce data movement**: More aggressive quantization, weight sharing, or skip computation
2. **Increase arithmetic intensity**: Batch multiple tokens (speculative decoding variant)
3. **Hardware changes**: More bandwidth (not available to us)

## Implementation Difficulty
N/A (analysis paper)
