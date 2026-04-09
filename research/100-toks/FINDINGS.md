# Speed Profiling Findings — 2025-04-09

## The Real Bottleneck

| Step | Time (ms) | % of 34ms/token |
|------|-----------|-----------------|
| Graph build | 0.3 | 1% |
| Graph alloc | 1.5 | 4% |
| Set inputs | 0.004 | 0% |
| graph_compute (async submit) | 1.5-2.5 | 5-7% |
| GPU execution | ~3 | 9% |
| **Unaccounted overhead** | **~27** | **79%** |

Graph reuse IS working after the first few tokens. GPU is fast (3ms/token, could do 323 tok/s).

The 79% overhead is in:
- `synchronize()` / GPU wait before logit extraction
- Sampling (common_sampler chain)
- KV cache bookkeeping
- Server slot management
- HTTP/SSE response streaming

## What We Tried
- Single command buffer (all nodes in main thread): **SLOWER** (-11.5%)
  - Serializes CPU encoding, preventing GPU overlap
  - Confirms async dispatch IS beneficial

## Key Insight
The problem is NOT:
- Metal dispatch overhead (GPU is 97% efficient within each token)
- Graph construction (0.3ms, graph is reused)
- Memory bandwidth (GPU runs at full speed when active)

The problem IS:
- Server-side per-token overhead (~27ms)
- Likely dominated by synchronize() + sampling

## Paths Forward (Revised)

### 1. Expert Deferral (top-6) — Still Valid
- Reduces GPU time from 3ms to ~2.3ms
- Also reduces server-side work (fewer expert results to manage)
- Expected: 28 → ~32 tok/s (modest but guaranteed)

### 2. Pipeline the Server Loop
- Don't synchronize after every token
- Sample from the PREVIOUS token's logits while computing the NEXT token
- This overlaps the 27ms server overhead with the next GPU compute
- Expected: potentially 2x (50-60 tok/s) if overlap is good

### 3. Batch Multiple Tokens
- Submit 2-4 tokens per decode call instead of 1
- Amortizes the 27ms server overhead across multiple tokens
- Expected: up to 2-4x if speculation acceptance rate is high

### 4. Bypass the Server Entirely
- Call llama.cpp directly (not through HTTP server)
- Eliminates HTTP + JSON + SSE overhead
- For LocalCode: possible since both are on the same machine
- Expected: could save 5-10ms per token
