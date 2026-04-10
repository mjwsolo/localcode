# Metal FlashAttention 2.0+ for Apple Silicon

- **Authors**: Philip Turner, Liu Liu (Draw Things)
- **Date**: 2024-2025 (v2.5 with Neural Accelerators for M5)
- **URL**: https://github.com/philipturner/metal-flash-attention
- **Blog**: https://engineering.drawthings.ai/

## Key Technique

Metal port of FlashAttention using tiled attention computation:
- Processes attention in blocks that fit in GPU SRAM (threadgroup memory)
- Avoids materializing the full NxN attention matrix
- Memory usage: O(N) instead of O(N^2)
- v2.5 leverages M5 Neural Accelerators for additional speedup

## Performance

- 43-120% faster inference vs non-FlashAttention
- 5-second 480p video generation on M5 iPad (16GB)
- v2.5: 4.6x improvement on M5 over M4

## Relevance to 64K Goal

**HIGH for enabling, not for compression**. FlashAttention doesn't reduce
KV cache size, but it makes attending over longer sequences feasible:
- At 64K tokens, naive attention needs to materialize a 64K x 64K matrix
  for global layers — impossible in limited memory
- FlashAttention processes this in tiles, needing only block-sized buffers
- Critical enabler: without FlashAttention, 64K context is compute-bound
  even if we solve the KV memory problem

llama.cpp already uses `-fa on` for FlashAttention. Need to verify the
Metal backend properly implements tiled attention for the 64K sequence
lengths we're targeting.

## Implementation Difficulty

**ALREADY DONE** (partially). llama.cpp has FlashAttention support via `-fa on`.
Verify it works correctly at 64K sequence length on Metal. May need to
tune tile sizes for optimal performance on M4 16GB.
