# KV Cache is 1 Bit Per Channel: Efficient LLM Inference with Coupled Quantization

- **Authors**: Tianyi Zhang, Jonah Yi, Zhaozhuo Xu, Anshumali Shrivastava
- **Date**: May 2024 (NeurIPS 2024)
- **URL**: https://arxiv.org/abs/2405.03917

## Key Technique

Coupled Quantization (CQ) exploits inter-channel dependencies in K/V embeddings.
Instead of quantizing each channel independently, CQ couples multiple channels
together. The joint entropy of multiple channels grows slower than the sum of
their marginal entropies, enabling 1-bit-per-channel encoding.

## Compression Ratio

- 1 bit per channel (theoretical 16x over FP16, ~4x over current turbo4)
- Achieves 1.4-3.5x throughput improvement over uncompressed baseline

## Quality Impact

Competitive with or outperforms existing baselines in preserving model quality.
Specific perplexity numbers vary by model; the key insight is that channel
coupling preserves information that per-channel quantization destroys.

## Relevance to 64K Goal

**HIGH**. If we can go from turbo4 (3.8x, ~355 MiB at 32K) to CQ-style 1-bit
encoding, the V cache alone would shrink ~4x further. A 64K context would
need roughly the same memory as current 32K with turbo4.

Math: 355 MiB (32K turbo4) -> with 1-bit V: ~90 MiB for V at 32K.
At 64K with 1-bit V + q8_0 K: ~180 MiB V + ~350 MiB K = ~530 MiB total.
Still tight but feasible within the ~2GB headroom.

## Implementation Difficulty

**HARD**. Requires:
1. Custom GGML quantization type (like turbo types but with channel coupling)
2. Metal kernel for coupled dequantization
3. Calibration step to learn channel groupings per layer
4. No existing llama.cpp support — would need to be built from scratch

Not a quick win, but the most impactful single technique for raw compression.
