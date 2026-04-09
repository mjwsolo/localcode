# Production-Grade Local LLM Inference on Apple Silicon

- **Authors**: (Multiple authors)
- **Date**: November 2025
- **URL**: https://arxiv.org/abs/2511.05502

## Key Findings

### Framework Throughput Ranking (M2 Ultra, 192GB)
1. MLX: ~230 tok/s
2. MLC-LLM: ~190 tok/s
3. llama.cpp: ~150 tok/s (short context)
4. Ollama: 20-40 tok/s
5. PyTorch MPS: ~7-9 tok/s

### llama.cpp Long-Context Degradation
llama.cpp drops from ~150 tok/s at short context to ~1.2 tok/s at 32K tokens due to lack of paged caching mechanisms. This is a massive degradation.

### MLX Advantage
MLX achieves highest sustained generation throughput through tighter Metal integration and optimized memory management.

### Metal-Specific Insights
- TTFT is compute-bound, decode is memory-bandwidth-bound
- Up to 546 GB/s bandwidth on M4 Max
- M4 base: ~120 GB/s
- Framework overhead is a significant contributor to the gap between theoretical and actual performance

## Relevance to 100 tok/s Goal
**HIGH** - Several important takeaways:

1. **MLX outperforms llama.cpp on Apple Silicon** - MLX's Metal integration is more mature. Our custom llama.cpp fork may be leaving performance on the table vs what MLX achieves.

2. **Framework overhead matters** - The gap between 150 tok/s (llama.cpp short ctx) and theoretical max suggests ~50% overhead from framework/kernel dispatch.

3. **Long-context degradation** - Our 32K context setting may be causing significant performance loss in llama.cpp. Need to investigate whether our TurboQuant KV cache mitigates this.

4. **Consider MLX backend** - For pure decode speed, MLX might be faster. But MLX doesn't support our TurboQuant KV cache or IQ3_S format.

## Implementation Difficulty
**MEDIUM** - The actionable insight is to profile our exact kernel dispatch overhead and Metal synchronization barriers. The paper suggests these are a major source of wasted cycles.
