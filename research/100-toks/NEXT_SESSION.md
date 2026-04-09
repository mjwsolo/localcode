# Next Session: GPU-Autonomous Decode Implementation

## What We Proved Today

| Test | Result |
|------|--------|
| Metal scheduling overhead | 0.41ms (not 35ms) |
| 512MB GPU read | 30ms (bandwidth bound) |
| Batched prompt eval | 146 tok/s |
| Sequential decode | 24 tok/s |
| 2 consecutive tokens | 47 tok/s (weights warm) |
| 4 consecutive tokens | 34 tok/s |
| 2 parallel sequences | 33 tok/s combined |
| Bridge argmax kernel | Compiles on Metal |
| ggml_argmax on Metal | EXISTS, works |
| ggml_get_rows on Metal | EXISTS, works |

## The Goal
Chain K forward passes in ONE compute graph using existing ggml ops:
`forward → argmax → get_rows → forward → argmax → get_rows → ...`

Target: 80-120 tok/s with K=4

## Exact Files to Modify

### 1. `src/models/gemma4-iswa.cpp` — Graph Builder
The constructor builds the forward pass graph. Modify to accept a `multi_step` parameter.
When K>1:
- After the logit output (line 246: `res->t_logits = cur`), add:
  - `ggml_argmax(logits)` → token_id
  - `ggml_get_rows(model.tok_embd, token_id)` → next embedding
  - `ggml_scale(embedding, sqrt(n_embd))` → scaled embedding
  - Run through all 30 layers again with position P+1
  - Repeat for K steps

### 2. `src/llama-context.cpp` — Decode Loop
In `process_ubatch()`:
- When doing single-token decode, set K=4 (configurable)
- Build the K-step graph
- After compute, extract K token IDs from the output
- Return all K tokens to the server

### 3. `tools/server/server-context.cpp` — Server Loop
- Handle K tokens per decode step instead of 1
- Feed K tokens into the sampling/history
- Handle EOS mid-batch

### Key Challenge: KV Cache Positions
Each step k writes KV at position P+k. The attention mask must allow
step k to see all positions [0..P+k]. This means the KV cache input
tensor needs K positions, and the attention mask needs to be causal
across all K steps.

Possible approach: set `ubatch.n_tokens = K` with positions [P, P+1, ..., P+K-1].
The forward pass already handles multi-token batches (prompt eval does this).
The only difference: token IDs for positions 1-K-1 come from GPU argmax,
not from CPU input.

### Working Branch
`experiment/gpu-autonomous-decode` on `/Users/marcsolomon/llama-cpp-turboquant/`
Working app is safe on `feature/turboquant-kv-cache`
