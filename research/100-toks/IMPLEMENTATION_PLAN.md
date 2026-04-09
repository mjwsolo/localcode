# GPU-Autonomous Multi-Step Decode — Implementation Plan

## The Innovation
Chain K forward passes in ONE Metal command buffer with GPU-side argmax bridge kernels.
No CPU roundtrip between tokens. Amortizes the 35ms per-commit overhead across K tokens.

K=4: 35ms / 4 = 8.75ms per token = 114 tok/s
K=8: 35ms / 8 = 4.4ms per token = 227 tok/s

BONUS: Weight pages loaded for step 1 are warm for steps 2-4. 
Second forward pass may cost only 5-10ms instead of 35ms.

## What We Verified (All Benchmarked)
- Metal scheduling: 0.41ms (NOT 35ms)
- 512MB GPU read: 30ms (bandwidth bound)
- Prompt eval (batched): 318 tok/s (proves batching works)
- Expert deferral: no help (bottleneck isn't expert count)
- Pre-fault pages: hurts on 16GB (swap pressure)
- Embedding on GPU: hurts (memory pressure)

## Implementation Phases

### Phase 1: Bridge Kernel (Week 1)
Write the Metal argmax + embedding lookup kernel.
- Input: logits buffer [vocab_size], embedding table (quantized)
- Output: next token ID, dequantized embedding vector
- Test standalone: verify it produces correct argmax

### Phase 2: Graph Chaining (Week 2)
Modify llama.cpp to build K-step graphs.
- In process_ubatch(): build K copies of the forward pass graph
- Insert bridge kernel nodes between each step
- Set KV cache positions to [P, P+1, ..., P+K-1]
- Submit as single graph → single Metal command buffer set

### Phase 3: Integration (Week 3)
Wire into the server decode loop.
- After K-step compute, read back K token IDs from step_tokens buffer
- Feed into sampling/history normally
- Handle EOS mid-batch (accept partial results)

### Phase 4: Optimization (Week 4)
- Adaptive K based on generation length
- Temperature/top-p sampling on GPU
- Quality benchmarking vs single-step baseline

## Key Files to Modify
- `ggml/src/ggml-metal/ggml-metal.metal` — add bridge_argmax_embed kernel
- `src/models/gemma4-iswa.cpp` — modify graph builder for K steps
- `src/llama-context.cpp` — process_ubatch loop for multi-step
- `tools/server/server-context.cpp` — decode loop to handle K tokens per step

## Risk Assessment
- Memory: bridge kernel adds negligible memory (<1KB)
- Quality: EXACT for greedy decode (argmax is deterministic)
- EOS waste: K-1 wasted steps worst case, but saves (K-1)*35ms
- KV cache: positions are deterministic, no issue

=== VALIDATED: 6x speedup with batching ===
Sequential decode: 24.5 tok/s (40.9 ms/tok)
Batched prompt eval: 146.4 tok/s (6.8 ms/tok)  
Speedup: 6.0x

GPU-autonomous K=4 target: 80-120 tok/s

