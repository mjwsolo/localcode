# TurboQuant turbo2: 2-bit KV Cache in llama.cpp

- **Authors**: Google Research (TurboQuant paper) + community implementations
- **Date**: 2025-2026 (ICLR 2026 for TurboQuant paper)
- **URL**: https://github.com/ggml-org/llama.cpp/discussions/20969
- **Code**: https://github.com/TheTom/llama-cpp-turboquant (feature/turboquant-kv-cache branch)

## Key Technique

TurboQuant turbo2 extends the existing TurboQuant family to 2-bit:
- **turbo2**: 2-bit, 6.4x compression, uses Walsh-Hadamard Transform (WHT)
- **turbo3**: 3-bit, 4.9x compression (our current fork supports this)
- **turbo4**: 4-bit, 3.8x compression (what we currently use for V cache)

WHT decorrelates values before quantization, reducing information loss at
extreme bit widths.

## Compression Ratio

- turbo2: 6.4x over FP16
- Compared to our turbo4 (3.8x): ~1.7x additional compression

## Quality Impact

- turbo2: +6.48% perplexity increase (vs turbo4: +0.23%)
- "Beat FP16 on perplexity alone but took a huge hit to KLD" — suggests
  the distribution shift is non-trivial even if aggregate PPL looks OK
- turbo3: negligible quality loss at 13B+ models

## Relevance to 64K Goal

**HIGHEST — IMMEDIATE PATH**. This is the lowest-hanging fruit because:
1. Our fork (mjwsolo/llama-cpp-turboquant) already has TurboQuant infrastructure
2. TheTom's fork already has turbo2 Metal support
3. We just need to change `-ctv turbo4` to `-ctv turbo2` (or turbo3)

Scenario A: q8_0-K + turbo2-V at 64K
- K cache: ~350 MiB (doubled from 32K)
- V cache: ~110 MiB (64K at 6.4x compression)
- Total: ~460 MiB — fits easily

Scenario B: q8_0-K + turbo3-V at 64K
- K cache: ~350 MiB
- V cache: ~145 MiB (64K at 4.9x compression)
- Total: ~495 MiB — fits comfortably

Scenario C: turbo3-K + turbo2-V at 64K
- K cache: ~145 MiB
- V cache: ~110 MiB
- Total: ~255 MiB — extreme savings, quality TBD

## Implementation Difficulty

**LOW**. Steps:
1. Pull turbo2/turbo3 support from TheTom's fork into our fork
2. Add turbo2/turbo3 as config options
3. Test quality on our IQ3_S Gemma 4 model at 64K
4. Benchmark decode speed (WHT adds compute but saves memory bandwidth)

Risk: turbo2's +6.48% PPL may be too much stacked on IQ3_S weights. turbo3
is the safer bet with 4.9x compression and negligible quality loss.
