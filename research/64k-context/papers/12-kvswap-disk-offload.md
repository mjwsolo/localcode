# KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference

- **Authors**: Huawei Zhang, Chunwei Xia, Zheng Wang
- **Date**: November 2025
- **URL**: https://arxiv.org/abs/2511.11907

## Key Technique

For devices with unified memory (CPU/GPU share RAM), offload KV cache to disk:
1. Store full KV cache on SSD
2. Use highly compact in-memory metadata to predict which entries to preload
3. Overlap computation with hardware-aware disk access
4. Tailored to specific storage characteristics (NVMe vs eMMC)

## Relevance to 64K Goal

**MEDIUM-LOW for latency, HIGH for capability**. Apple Silicon has fast NVMe
(~7 GB/s read on M4). At 64K context:
- Full FP16 KV cache: ~1.4 GB on disk
- Turbo4 compressed: ~710 MiB on disk
- NVMe read latency: ~100 MiB in ~14ms

The latency hit depends on how much KV needs to be fetched per token.
If combined with ShadowKV-style sparse selection (only 1.56% of tokens),
we'd fetch ~11 MiB per decode step — ~1.6ms from NVMe.

This is a viable fallback for truly long contexts (128K+) where in-memory
KV cache simply won't fit, but adds decode latency.

## Implementation Difficulty

**MEDIUM**. Apple Silicon NVMe is fast and unified memory simplifies the
design. Main challenges:
1. Efficient async I/O to overlap with compute
2. Metadata structure for predicting needed entries
3. Integration with llama.cpp KV cache management
