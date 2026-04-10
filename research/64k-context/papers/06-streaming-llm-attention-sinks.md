# StreamingLLM: Efficient Streaming Language Models with Attention Sinks

- **Authors**: Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis (MIT + Meta)
- **Date**: September 2023 (ICLR 2024)
- **URL**: https://arxiv.org/abs/2309.17453
- **Code**: https://github.com/mit-han-lab/streaming-llm

## Key Technique

LLMs allocate massive attention to the first few tokens ("attention sinks")
regardless of their semantic importance. StreamingLLM keeps:
1. A few "sink" tokens (first 4 tokens typically)
2. A sliding window of recent tokens

This enables infinite-length generation with fixed memory, up to 22.2x speedup
over sliding window recomputation.

## Memory Model

Fixed memory: sink_tokens + window_size. Independent of total sequence length.
Typical: 4 sink + 4096 window = constant ~4100 token KV cache regardless of
whether you've processed 10K or 10M tokens.

## Quality Impact

Works well for streaming/generation tasks. However, **cannot retrieve
information from evicted tokens** — only useful when you don't need to look
back beyond the window. Not suitable for tasks requiring full context recall.

## Relevance to 64K Goal

**LOW-MEDIUM for our use case**. A coding assistant needs to recall code from
earlier in the conversation. Pure streaming eviction would lose that context.

However, the attention sink insight is valuable:
- Always keep first few tokens in KV cache (system prompt)
- Combine with DuoAttention: streaming heads use StreamingLLM pattern,
  retrieval heads keep full cache
- Gemma 4's 26 sliding window layers already implement this pattern

The real value is as a component of a hybrid strategy, not standalone.

## Implementation Difficulty

**LOW**. StreamingLLM is already well-understood and the sliding window concept
is built into Gemma 4. The attention sink preservation is a simple
implementation detail (never evict first N tokens).
