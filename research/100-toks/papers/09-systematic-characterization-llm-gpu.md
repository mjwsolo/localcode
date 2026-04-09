# A Systematic Characterization of LLM Inference on GPUs

- **Authors**: (Multiple authors)
- **Date**: December 2024
- **URL**: https://arxiv.org/abs/2512.01644

## Key Findings

### MoE vs Dense Bandwidth Utilization
- Dense FFN arithmetic intensity: 15.74 FLOP/byte (at larger batches)
- MoE FFN arithmetic intensity: 8 FLOP/byte (stays low regardless of batch)
- MoE SM utilization: 28-34% with high DRAM pressure (>80%)
- Dense SM utilization: scales 41.7% -> 76.5%

### Why MoE Wastes Bandwidth
MoE routing distributes the global batch across experts, creating small per-expert effective batches. Even with batch=64, each of 128 experts might see 0-4 tokens. This prevents computational intensity from reaching dense-model levels.

### Decode Phase Specifics
- Decode has 48-56% higher DRAM utilization than prefill
- Per-token GEMVs reload weights from DRAM every time
- KV-cache footprint grows linearly, intensifying bandwidth pressure

## Relevance to 100 tok/s Goal
**CRITICAL** - This paper explains our exact bottleneck. For single-batch decode with MoE:
- Each expert computes a matrix-vector product (GEMV)
- Arithmetic intensity is ~1 FLOP/byte (even lower than the paper's batched scenarios)
- We're fundamentally memory-bandwidth-bound
- The ONLY way to go faster is to reduce bytes read per token

### Paths Forward (from this analysis)
1. **Weight-only quantization on FFN layers**: We're already at IQ3_S, further compression trades quality
2. **KV-cache quantization**: We have TurboQuant (turbo4-V), already done
3. **Reduce expert reads**: Skip/defer experts, cache hot experts, or batch multiple tokens

## Implementation Difficulty
N/A (characterization paper) - but provides the theoretical framework for all our optimization decisions.

## Critical Number
At IQ3_S (~3.5 bits/weight), our 3.8B active weights = ~1.2GB per token.
At 120 GB/s bandwidth = theoretical 100 tok/s.
28% utilization = 28 tok/s (exactly what we measure).
To reach 100 tok/s, we need ~100% bandwidth utilization OR reduce active weight reads.
