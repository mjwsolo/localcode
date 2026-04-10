# ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- **Authors**: Hanshi Sun, Li-Wen Chang, Wenlei Bao, et al. (ByteDance)
- **Date**: October 2024 (ICML 2025)
- **URL**: https://arxiv.org/abs/2410.21465
- **Code**: https://github.com/bytedance/ShadowKV

## Key Technique

Offloads V cache to CPU while keeping compressed K cache landmarks on GPU:
1. Post-RoPE key cache exhibits spatial locality (adjacent tokens are similar)
2. Store mean values as "landmarks" for token chunks
3. During decode, use landmarks to select important KV pairs (only 1.56%)
4. Reconstruct sparse KV pairs on-the-fly from CPU

## Compression Ratio

6x larger batch sizes, 3.04x throughput on A100. The landmark approach
means GPU only holds compressed K summaries + small active V subset.

## Relevance to 64K Goal

**MEDIUM**. The CPU offloading concept is relevant for unified memory Apple
Silicon, but the distinction between GPU VRAM and CPU RAM doesn't apply in the
same way — both share the same physical memory pool.

However, the **landmark-based sparse retrieval** idea is powerful:
- Store full KV on disk (SSD) or compressed in memory
- Keep only landmarks (chunk means) in hot memory
- Reconstruct on demand during attention

For Gemma 4's 4 global layers at 64K, this could mean:
- Store landmarks for 64K tokens (very small)
- Only fetch ~1000 important KV pairs per layer per decode step
- Massive effective compression with minimal quality loss

## Implementation Difficulty

**HARD**. Requires:
1. Landmark computation during prefill
2. Sparse KV selection during decode (changes attention kernel)
3. On-demand KV reconstruction from compressed store
4. Custom Metal kernels for sparse attention
5. The 1.56% selection rate needs validation on Gemma 4 GQA heads
