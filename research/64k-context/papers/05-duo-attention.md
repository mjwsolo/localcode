# DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

- **Authors**: Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, Song Han (MIT Han Lab)
- **Date**: October 2024 (ICLR 2025)
- **URL**: https://arxiv.org/abs/2410.10819
- **Code**: https://github.com/mit-han-lab/duo-attention

## Key Technique

Attention heads fall into two categories:
- **Retrieval heads**: Need full KV cache across all tokens (critical for
  long-range information access)
- **Streaming heads**: Only attend to recent tokens + attention sinks
  (can use a tiny sliding window cache)

A lightweight optimization-based algorithm with synthetic data identifies
which heads are which. Only retrieval heads get full KV cache; streaming
heads get constant-size cache regardless of context length.

## Compression Ratio

- MHA models: 2.55x memory reduction
- GQA models: 1.67x memory reduction
- Combined with quantization: enables 3.3M context on A100

## Quality Impact

Outperforms H2O, TOVA, StreamingLLM on Needle-in-a-Haystack and LongBench
with same KV budget. Minimal accuracy loss.

## Relevance to 64K Goal

**HIGH — complementary to quantization**. This is orthogonal to turbo2/turbo3.

Gemma 4 26B uses GQA (8 query heads, 4 KV heads) with hybrid attention:
- 26 sliding window layers (1024 token window) — already streaming!
- 4 global attention layers — these need full KV cache

This is a natural fit: Gemma 4's architecture already implements a form of
DuoAttention. The 26 sliding layers only need 1024-token KV windows.
Only the 4 global layers need full-context KV cache.

**Actual KV budget at 64K with Gemma 4 architecture**:
- 26 sliding layers: 1024 tokens x 4 KV heads x 256 head_dim x 2 (K+V) = small constant
- 4 global layers: 64K tokens x 4 KV heads x 512 head_dim x 2 (K+V) = bulk of memory

The sliding window layers contribute ~5% of total KV cache. The global layers
with their larger head_dim (512 vs 256) dominate. DuoAttention-style analysis
could identify which of the 4 global layers truly need full context, potentially
halving the global cache.

## Implementation Difficulty

**MEDIUM**. Gemma 4 already has the sliding/global split built in. The question
is whether llama.cpp properly implements the 1024-token sliding window eviction
for the 26 local layers. If it does, most KV savings are already captured.
If it allocates full context for all layers, this is a huge win to fix.

Steps:
1. Verify llama.cpp KV allocation for Gemma 4 sliding vs global layers
2. If sliding layers get full allocation, implement proper window eviction
3. Profile which global layers are retrieval-critical
4. Consider smaller windows for non-critical global layers
