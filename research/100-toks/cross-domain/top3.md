# Top 3 Cross-Domain Solutions

## Selection Criteria
Ranked by total score (feasibility + novelty + impact), with tiebreakers favoring higher impact.

---

## #1: GPU-Driven Autoregressive Loop (from Game Engine GPU-Driven Rendering + io_uring)

**Score**: Feasibility 7, Novelty 10, Impact 10 = **27**

### Original Problem (Game Engines)
Modern game engines must render millions of objects per frame. Each CPU-submitted draw call costs driver overhead. At 10K+ objects, the CPU becomes the bottleneck, not the GPU. The solution: GPU-driven rendering. The CPU uploads ALL potential draw commands once. A GPU compute shader decides which to execute. One `drawIndirect` call replaces thousands of individual draw calls. The CPU is removed from the per-frame loop.

### Original Problem (io_uring)
Each Linux `read()`/`write()` requires a user-kernel context switch. At millions of IOPS (NVMe drives), this dominates. io_uring batches N operations into a single submission queue entry, and the kernel processes them all from one `io_uring_enter()` call. Results appear in a completion ring without further syscalls.

### Mapping to Our Problem
Our CPU submits one "draw call" (token decode) per generated token. The 35ms Metal scheduling overhead is the "driver overhead" that game engines eliminated. io_uring's insight adds: the RESULTS should also avoid per-item overhead (no synchronize per token).

The combined solution: **move the entire autoregressive loop to the GPU**. The CPU submits one command buffer containing an entire multi-token decode loop. The GPU runs: decode -> sample -> update KV -> decode -> sample -> ... All intermediate tokens stay on GPU. The CPU only reads back the final sequence.

### Why This Works
The 35ms overhead occurs per command buffer commit. If we put 16 token decodes in ONE command buffer, overhead drops to 35ms/16 = 2.2ms per token = **~450 tok/s theoretical**. Even if the on-GPU sampling adds overhead, we'd likely hit memory bandwidth limits (~100 tok/s) before scheduling limits.

### Implementation Path

**Phase A: GPU-side argmax kernel (1-2 days)**
- Write a Metal compute shader that reads the logits buffer (vocab_size = 262144 for Gemma) and outputs the argmax token ID
- This replaces CPU-side `llama_sampler_sample()` for greedy decoding
- Token ID written to a GPU buffer, no CPU readback needed

**Phase B: Token embedding lookup on GPU (1 day)**
- Write a Metal compute shader that reads the token ID from argmax output and copies the corresponding embedding row from the embedding weight matrix
- Output: the input tensor for the next decode iteration, entirely on GPU

**Phase C: Multi-token command buffer (1 week)**
- Encode N iterations of the decode graph into ONE command buffer
- Between iterations, insert the argmax + embedding lookup kernels
- Use `MTLFence` to sequence operations within the buffer
- KV cache positions are pre-computed (they increment linearly)
- The only CPU work: update the KV cache position offsets (known in advance)

**Phase D: EOS detection and early termination (2-3 days)**
- Add a small compute kernel after argmax that checks if token == EOS
- If EOS detected, write to a shared `MTLBuffer` with `MTLStorageModeShared`
- CPU polls this buffer (no synchronize needed — unified memory)
- For early termination: use `MTLIndirectCommandBuffer` with variable execution range

### Risks and Mitigations
1. **Top-k/top-p sampling on GPU**: Argmax (greedy) is trivial. Full sampling (temperature, top-k, top-p, repetition penalty) requires a sort on GPU. Mitigation: start with greedy, add GPU-side top-k later (parallel reduction is well-studied).
2. **KV cache management**: KV cache offsets must be known when encoding the command buffer. For autoregressive decode, they increment by 1 per token — this is known in advance. No problem.
3. **Command buffer size**: Encoding 16 decode iterations might exceed Metal's command buffer limits. Mitigation: benchmark with increasing N to find the sweet spot.
4. **Debugging difficulty**: GPU-side loops are hard to debug. Mitigation: keep the CPU-side path as fallback, use Metal GPU capture for debugging.

### Expected Performance
- Current: 35ms overhead / 1 token = 28 tok/s
- With N=8: 35ms overhead / 8 tokens + 8 * ~0.5ms GPU compute = ~200 tok/s
- With N=16: 35ms overhead / 16 tokens + 16 * ~0.5ms GPU compute = ~140 tok/s (diminishing returns from GPU compute time)
- Conservative estimate: **80-120 tok/s** (accounting for GPU sampling overhead and memory bandwidth)

---

## #2: Branch-Parallel Speculation (from HFT Market Making)

**Score**: Feasibility 7, Novelty 8, Impact 9 = **24**

### Original Problem (HFT Market Making)
Market makers must provide continuous quotes on an exchange. Updating a quote requires a round-trip to the exchange (5-50us). If the market moves faster than the round-trip, the quote goes stale and the market maker loses money. Solution: pre-compute quotes for MULTIPLE price levels and submit all at once. When the market moves to price P, the quote at P is already live. Wasted compute on unused quotes is cheaper than latency.

### Mapping to Our Problem
After generating token T, we don't know token T+1. But we know the TOP-K candidates. With K=4, one of the top-4 tokens will be T+1 about 85-95% of the time (Gemma 4's distribution is quite peaked for code).

**Key insight from ROOT_CAUSE.md**: GPU compute time is 0ms per token, but scheduling overhead is 35ms. This means running a batch of 4 tokens costs the SAME 35ms as running 1 token. We get 4 speculative paths for free.

### How It Works
1. Generate token T, get logits
2. Sample token T+1, AND record top-4 candidates: {c1, c2, c3, c4}
3. Submit ONE batched forward pass with batch_size=4, computing the full decode for all 4 candidates simultaneously
4. When token T+1 is confirmed (it's one of c1-c4), the KV cache for that path is already computed
5. Discard the other 3 KV cache entries
6. Repeat: sample T+2 from the correct path's logits, take top-4 of T+2, batch-decode all 4

### Why This Is Different From Standard Speculative Decoding
- Standard spec decode: predict a SEQUENCE of N tokens, verify in one pass. Fails when early predictions are wrong.
- Branch-parallel: predict ALTERNATIVES for the NEXT token only. Much higher hit rate (top-4 covers 85-95%). No wasted sequential speculation.
- Standard spec decode with a draft model: requires 2 models = 2x 35ms overhead = slower. Branch-parallel uses the SAME model with batch_size=4, ONE 35ms overhead.

### Implementation Path

**Phase A: Batched single-token decode (2-3 days)**
- Modify llama.cpp to support "branching" KV cache: 4 parallel KV cache slots diverging from the same prefix
- Submit batch of 4 tokens in one forward pass
- This already works in llama.cpp's prompt eval path (batch size > 1)

**Phase B: Top-K path selection (1 day)**
- After sampling token T+1, record top-4 candidates
- Feed all 4 as a batch into the next decode call
- After decode, check which candidate was actually selected
- Keep that KV cache slot, discard others

**Phase C: KV cache slot management (3-4 days)**
- Need efficient "fork and discard" for KV cache
- Copy the current KV state to 4 slots (or use copy-on-write)
- After selection, promote the winning slot and free the others
- Memory cost: 4x KV cache for one layer of speculation = 4 * 355MB = 1.4GB. Too much for 16GB.
- Optimization: only branch the TOP layer's KV (the rest is shared prefix). Cost: 4 * 355MB / 26 layers * 1 layer = ~55MB. Acceptable.

**Wait — there's a problem**: we can't just branch one layer. Each layer's KV cache depends on the token processed. We need all 26 layers' KV for each branch.

**Resolution**: Use a smarter memory scheme. The first 25 layers of KV are identical for the shared prefix. Only the LAST decoded position (1 token) differs across branches. Store delta: 4 branches * 1 token * 26 layers * 2 (K+V) * head_dim. For Gemma 4: 4 * 1 * 26 * 2 * 128 * 32 heads * 2 bytes = ~1.3MB. Totally fine.

### Expected Performance
- Hit rate: ~90% (top-4 covers next token 90% of the time for code)
- When hit: 1 commit for 1 real token (same as now, but KV is pre-computed for the NEXT token too)
- Effective: eliminates one full commit cycle per token
- Current: 35ms/token = 28 tok/s
- With branch-parallel: every 2 real tokens costs 2 commits (one for branching, one for next branch) but the second commit's KV is pre-computed. Effective: 35ms/1.9 tokens = ~53 tok/s
- If combined with #1 (GPU-side sampling): branch 4 ways AND run 4 iterations on GPU = **~100 tok/s**

---

## #3: Persistent Indirect Command Buffer (from HTTP/2 Multiplexing + Triple Buffering)

**Score**: Feasibility 8, Novelty 8, Impact 9 = **25**

### Original Problem (HTTP/2)
HTTP/1.1 creates a new TCP connection per request. Connection setup (TCP handshake + TLS) costs 100-300ms. HTTP/2 multiplexes all requests over ONE persistent connection. Setup happens once.

### Original Problem (Triple Buffering)
Games pre-allocate 3 frame buffers. The GPU always has a buffer to render into. The CPU always has a buffer to prepare. No allocation or setup per frame — everything is pre-allocated and rotated.

### Mapping to Our Problem
Every token decode creates a NEW command buffer, encodes ALL operations into it, commits it, and waits. The 35ms includes: command buffer creation, resource validation, GPU scheduling setup, and kernel dispatch preparation. Most of this is IDENTICAL between tokens — the graph structure, buffer bindings, kernel pipelines, and threadgroup sizes never change.

The HTTP/2 + triple buffer insight: **create the command buffer ONCE and re-execute it**.

### How It Works
Metal provides `MTLIndirectCommandBuffer` (ICB), designed for exactly this pattern:
1. Pre-encode all compute dispatches into an ICB once
2. Each frame/token, update only the CHANGED arguments (token ID, KV offset)
3. Execute the ICB — Metal skips most validation because the structure hasn't changed
4. The ICB persists across executions

### Implementation Path

**Phase A: Benchmark ICB overhead (2 days)**
- Create a minimal test: encode a simple matmul into an ICB
- Execute it 1000 times, measure per-execution overhead
- Compare against regular command buffer commit overhead
- If ICB execution < 5ms: proceed. If still 35ms: the overhead is elsewhere.

**Phase B: ICB-compatible graph encoding (1-2 weeks)**
- llama.cpp's Metal backend encodes operations using `MTLComputeCommandEncoder`
- ICBs use `MTLIndirectComputeCommand` — different API but same concepts
- Modify `encode_async` in `ggml-metal-context.m` to encode into an ICB
- First execution: full encode. Subsequent executions: only update argument buffers.

**Phase C: Argument buffer patching (3-4 days)**
- Identify which buffers change between tokens:
  - Input token embedding (changes every token)
  - KV cache write offset (increments by 1)
  - Attention mask (extends by 1 position)
- Create "argument buffers" for these and update them CPU-side between executions
- All other bindings (weight buffers, pipeline states) stay fixed

**Phase D: Double-buffer the ICB (2-3 days)**
- While ICB-A executes on GPU, CPU patches arguments for ICB-B
- Alternate between A and B each token
- Overlap CPU argument patching with GPU execution

### Risks and Mitigations
1. **ICB limitations**: ICBs have restrictions on what can be encoded. Some ggml operations might not be ICB-compatible. Mitigation: identify incompatible operations and keep them in a regular command buffer; only ICB-encode the hot path.
2. **Argument buffer complexity**: Updating scattered arguments across hundreds of kernels is error-prone. Mitigation: use a single "parameters" buffer that all kernels reference, update it in one memcpy.
3. **Metal driver behavior**: Apple's documentation says ICBs "amortize encoding cost" but doesn't guarantee reduced scheduling overhead. Mitigation: Phase A benchmark will tell us immediately if this helps.

### Expected Performance
- If ICB eliminates the 35ms scheduling overhead:
  - Per-token cost = GPU compute only (~0.5ms for single token) + ICB execution overhead
  - Theoretical: ~500+ tok/s
  - Realistic (memory bandwidth limited): ~100 tok/s
- If ICB reduces overhead by 50% (17ms):
  - Per-token cost = 17ms + 0.5ms = 17.5ms
  - ~57 tok/s
- Conservative estimate: **50-100 tok/s**
- **Highest feasibility** of the three because ICBs are a supported Metal feature designed for this exact use case.

---

## Combined Strategy

The three solutions are complementary:

| Phase | Solution | Expected Speed | Effort |
|-------|----------|---------------|--------|
| 1 | #3: Persistent ICB (reduce 35ms overhead) | 50-60 tok/s | 2-3 weeks |
| 2 | #2: Branch-parallel (4x batching for free) | 70-80 tok/s | 1-2 weeks |
| 3 | #1: GPU-driven loop (eliminate CPU roundtrip) | 100+ tok/s | 2-3 weeks |

Start with #3 (ICB) because it has the highest feasibility and immediately tells us how much of the 35ms is Metal validation (reducible) vs. something else. Then add branch-parallel on top. Finally, move the full loop to GPU.

**Critical first experiment**: Phase A of #3 — benchmark ICB execution overhead. If it's still 35ms, the overhead is in the Metal driver itself (not validation), and we pivot entirely to #1 (GPU-driven loop to amortize the fixed cost).
