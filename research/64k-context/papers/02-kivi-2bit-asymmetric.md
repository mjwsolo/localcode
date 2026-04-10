# KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- **Authors**: Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu
- **Date**: February 2024 (ICML 2024)
- **URL**: https://arxiv.org/abs/2402.02750
- **Code**: https://github.com/jy-yuan/KIVI

## Key Technique

Asymmetric 2-bit quantization with different strategies for K and V:
- **Keys**: Quantized per-channel (columns have different distributions)
- **Values**: Quantized per-token (rows have different distributions)

This asymmetry is critical — K and V have fundamentally different activation
patterns. Tuning-free: no calibration data needed.

## Compression Ratio

2-bit for both K and V = 8x compression over FP16.
Compared to our current q8_0-K + turbo4-V (~3.8x average), this would be ~2x
further compression.

## Quality Impact

- 2.6x less peak memory (including model weights)
- Maintains "almost the same quality" on Llama, Falcon, Mistral
- Enables 4x larger batch size

## Relevance to 64K Goal

**VERY HIGH**. This is the most directly applicable paper.

Current: q8_0-K + turbo4-V at 32K = 355 MiB
KIVI-style: 2-bit K (per-channel) + 2-bit V (per-token) at 32K = ~90 MiB
At 64K: ~180 MiB — easily fits with massive headroom.

The question is quality. Our model is already IQ3_S quantized, so stacking
2-bit KV on top of 3-bit weights may compound errors. Needs testing.

## Implementation Difficulty

**MEDIUM**. The per-channel K / per-token V pattern could potentially be
implemented as a new GGML type. The TurboQuant fork already has the
infrastructure for custom KV types. Key challenges:
1. Per-channel K quantization requires knowing full channel statistics
   (need a running calibration during prefill)
2. Per-token V is straightforward (similar to existing quantization)
3. Metal kernels needed for dequantization
4. The asymmetric approach aligns well with existing q8_0-K / turbo-V split
