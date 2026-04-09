# Novel Approaches to Eliminate the 35ms Metal Overhead

Ranked by promise, from most to least feasible.

---

## 1. SPECULATIVE MULTI-TOKEN COMMAND BUFFER (Rank: HIGHEST)

### The Idea
Instead of generating one token per Metal submission, encode N autoregressive decode steps into a SINGLE command buffer, where each step reads the previous step's output via argument buffers / indirect dispatch. The GPU executes all N steps without returning to the CPU.

### First-Principles Reasoning
The 35ms overhead is per-submission. If we submit 4 tokens worth of work in one command buffer, we pay 35ms once and get 4 tokens -- effectively 8.75ms/token = 114 tok/s.

But wait: autoregressive decoding is sequential. Token N+1 depends on the argmax of token N's logits. How can we pre-encode the next step if we don't know the input token?

**Key insight**: We don't need to know the EXACT next token. We can:

1. **Embed ALL tokens in the vocabulary** (or the top-K) as a lookup table in GPU memory
2. After the forward pass produces logits, a GPU-side kernel does argmax and uses the result as an INDEX into the pre-computed embedding table
3. The next forward pass reads from that indexed position

Metal supports **indirect command buffers** (ICBs) since Metal 2 / Apple GPU Family 4. ICBs allow the GPU to encode its OWN commands based on runtime data. But even without ICBs, we can use a simpler approach:

**Argument Buffers + Indirect Addressing**: Store all token embeddings in a single MTLBuffer. The GPU writes the argmax token ID to a shared buffer. The next layer reads the embedding by indexing: `embedding[token_id * dim]`. This is just a gather operation -- no CPU involvement needed.

### What Needs to Be Built
1. **Pre-computed embedding table**: All 256K token embeddings (256K * 3584 * sizeof(IQ3_S) ~= 300MB dequantized to fp16, or we dequantize on-the-fly from the GGUF) stored as a single MTLBuffer
2. **GPU-side argmax kernel**: Takes logits buffer, writes token_id to a small MTLBuffer
3. **GPU-side embedding lookup**: Reads token_id, gathers the embedding vector
4. **Modified graph builder**: Encodes N sequential forward passes into one command buffer, with the embedding lookup between each pass
5. **KV cache management**: Each sub-step must write to the correct KV cache position

### Feasibility: HIGH
- No new Metal features needed beyond what's already used
- Argument buffers are well-supported on Apple Silicon
- The embedding table is just the model's `token_embd` tensor -- already in GGUF
- GPU-side argmax is trivial (reduction kernel)
- The hard part is modifying llama.cpp's graph builder to chain multiple forward passes

### Expected Impact: 3-4x throughput
- 4 tokens per submission: 35ms / 4 = 8.75ms/tok = 114 tok/s
- 8 tokens per submission: 35ms / 8 = 4.4ms/tok = 227 tok/s (if GPU compute stays low)
- Diminishing returns as GPU compute becomes the bottleneck

### Why Nobody Has Done This
- llama.cpp's architecture assumes one graph_compute per token
- The graph builder creates a new graph each time (or reuses one-step graphs)
- It requires deep changes to the model builder, not just the Metal backend
- Most platforms (CUDA) don't have this overhead, so the motivation doesn't exist

### Risk
- Sampling strategies beyond greedy (temperature, top-p, top-k) become harder on GPU
- If we speculatively pick the wrong token... wait, we're NOT speculating. We do the REAL argmax on GPU. This is exact for greedy decoding.
- For non-greedy: we'd need GPU-side sampling, which is possible but complex (GPU random number generation + cumulative probability scan)

---

## 2. PERSISTENT GPU KERNEL WITH SHARED MEMORY MAILBOX (Rank: HIGH)

### The Idea
Launch a single, never-ending Metal compute kernel at startup. This kernel spins on a shared memory location (the "mailbox"), waiting for instructions. The CPU writes the next token's embedding to shared memory, sets a flag, and the GPU kernel picks it up and runs the full forward pass without any command buffer submission.

### First-Principles Reasoning
The 35ms comes from the submit-wait pipeline. What if we never submit? Apple Silicon's unified memory means both CPU and GPU can read/write the same physical memory. If a kernel is already running, it can poll a memory location for new work.

### Implementation Sketch
```metal
kernel void persistent_decode(
    device atomic_uint * mailbox,
    device float * input_embedding,
    device float * output_logits,
    device float * weights,
    device float * kv_cache,
    ...
) {
    while (true) {
        // Spin-wait for CPU to signal new work
        while (atomic_load_explicit(mailbox, memory_order_acquire) == 0) {
            // GPU spins here -- costs power but no latency
        }

        // Read input, run forward pass
        forward_pass(input_embedding, output_logits, weights, kv_cache);

        // Signal completion
        atomic_store_explicit(mailbox, 0, memory_order_release);
        atomic_store_explicit(mailbox + 1, 1, memory_order_release); // "done" flag
    }
}
```

The CPU side:
```c
// Write embedding
memcpy(shared_input, embedding, dim * sizeof(float));
// Signal GPU
atomic_store(mailbox, 1);
// Wait for result
while (atomic_load(mailbox + 1) == 0) { /* spin */ }
// Read logits
memcpy(logits, shared_output, vocab_size * sizeof(float));
```

### Feasibility: MEDIUM-LOW
**Critical problem**: Metal does NOT support infinite-duration kernels. The GPU watchdog timer will kill any kernel that runs longer than ~5-10 seconds. There is no public API to disable this.

**Workaround**: Use `MTLCommandBuffer`'s completion handler to immediately re-submit the persistent kernel when it's about to time out. But this reintroduces the submission overhead periodically.

**Another problem**: A single Metal compute kernel cannot do everything a forward pass needs. A forward pass requires hundreds of different kernels (matmul, softmax, RoPE, SiLU, etc.) with different thread configurations. You'd have to implement the ENTIRE forward pass as ONE monolithic kernel with manual thread synchronization -- effectively writing a GPU-side scheduler.

**Apple Silicon atomics**: `device` memory atomics ARE supported on Apple GPUs (Metal 2.0+). Both CPU and GPU can do atomic operations on shared memory. The coherence model should work.

### Expected Impact: Potentially 10x if feasible
- Eliminates ALL submission overhead
- Latency would be: CPU write time + GPU compute + CPU read time = ~3-5ms/token = 200-300 tok/s theoretical

### Why Nobody Has Done This
- Metal watchdog timer prevents truly persistent kernels
- Implementing a full transformer forward pass in one kernel is a massive engineering effort
- No existing framework supports this pattern
- The monolithic kernel would lose all the optimized per-operation kernels that llama.cpp has

---

## 3. DOUBLE-BUFFERED PIPELINE PARALLELISM (Rank: HIGH)

### The Idea
Overlap the CPU-side work (encoding the next command buffer, setting inputs) with the GPU executing the current token. Use two sets of buffers and alternate. The CPU is always one token ahead of the GPU.

### First-Principles Reasoning
Looking at the 35ms breakdown:
- ~5ms is CPU-side encoding
- ~12ms is page table / IOMapper validation
- ~5ms is GPU compute
- ~13ms is scheduling + signaling overhead

If we pipeline: while the GPU processes token N, the CPU encodes token N+1's command buffer. When GPU finishes N, we immediately commit N+1 (which is already encoded). This eliminates the sequential encoding time.

### What Needs to Be Built
1. **Double-buffered KV cache**: Two copies of the KV cache (or a ping-pong scheme)
2. **Async command buffer encoding**: While waiting for GPU, encode the next graph
3. **Pre-committed command buffer**: Use `[cmd_buf enqueue]` early, then `[cmd_buf commit]` the instant the previous one completes

### Feasibility: HIGH
- llama.cpp already has `pipeline_parallel` support in the code
- Metal's command queue naturally supports this -- enqueue multiple command buffers and they execute in order
- The graph reuse mechanism already caches the graph structure

### Expected Impact: ~30-40% improvement
- Saves the CPU encoding time (~5ms) but NOT the per-submission overhead
- Would get us from 28 tok/s to maybe 36-40 tok/s
- Not enough for 100 tok/s alone, but stacks with other approaches

### Why It's Limited
- The dominant cost is the per-submission overhead (scheduling, page table validation), not CPU encoding
- Double-buffering the KV cache doubles memory usage (355 MiB -> 710 MiB), which is tight on 16GB

---

## 4. PRE-FAULT + WIRE ALL MMAP PAGES (Rank: MEDIUM-HIGH)

### The Idea
Before the first decode, read every page of the mmap'd GGUF file to force all pages into physical memory, then use `mlock()` or `vm_wire()` to prevent them from being evicted. This eliminates the page fault and IOMapper validation costs.

### First-Principles Reasoning
If the 10.4GB mmap pages are ALL resident and wired, then Metal's page table validation becomes a no-op (or near-no-op) because all entries are already valid. The IOMapper doesn't need to fault-in any pages.

### Implementation
```c
// At startup, after mmap:
void * model_data = mmap(..., 10.4GB, ...);

// Pre-fault every page
volatile char * p = (volatile char *)model_data;
for (size_t i = 0; i < model_size; i += PAGE_SIZE) {
    (void)p[i];  // force page fault now
}

// Wire pages (requires root or entitlement)
mlock(model_data, model_size);
```

### Feasibility: MEDIUM
- `mlock()` on 10.4GB would wire 10.4GB of the 16GB total -- leaving only 5.6GB for everything else (OS, apps, KV cache). This might cause the system to swap aggressively elsewhere.
- macOS has a process limit on mlock'd memory (usually much less than 10GB)
- The `sysctl iogpu.wired_limit_mb=14336` trick already pushes GPU memory limits
- Pre-faulting is easy; keeping pages resident under memory pressure is the hard part

### Expected Impact: 5-15ms reduction (maybe)
- If page table validation is truly 8-15ms of the 35ms, and we eliminate it, we'd get 20-27ms/token = 37-50 tok/s
- But this is speculative -- the IOMapper overhead might be smaller than estimated

### What Would Confirm This
- Profile with `Instruments.app` using Metal System Trace to see if page faults occur during command buffer commit
- Compare decode speed when the model was just fully read (all pages warm) vs. after sitting idle (pages potentially evicted)

---

## 5. FUSED MEGA-KERNEL: ONE DISPATCH PER LAYER (Rank: MEDIUM)

### The Idea
Instead of 10-15 separate kernel dispatches per transformer layer (Q/K/V proj, RoPE, attention, softmax, output proj, gate, expert select, 8x expert FFN, combine), fuse them into ONE kernel dispatch per layer. This reduces the number of dispatches from ~400 to ~30, reducing per-dispatch overhead within the command buffer.

### First-Principles Reasoning
Even within a single command buffer, each `dispatchThreadgroups` call has overhead:
- Pipeline state switch
- Memory barrier
- Threadgroup setup

If Metal's per-dispatch overhead within a command buffer is ~50-100us, then 400 dispatches = 20-40ms of intra-buffer overhead. This could explain a significant chunk of the 35ms.

But wait -- during prompt eval, we also have ~400 dispatches and achieve 318 tok/s. So intra-buffer dispatch overhead is NOT the dominant factor. This approach would help, but it's not the silver bullet.

### Feasibility: LOW-MEDIUM
- Writing fused Metal shaders for an entire transformer layer is an enormous engineering effort
- Different layers have different expert selections, so the fused kernel needs dynamic dispatch
- MoE routing makes fusion especially hard because the active experts vary per token
- llama.cpp's existing per-op kernels are heavily optimized; a fused version might be slower due to register pressure

### Expected Impact: 10-20% improvement at best
- Reduces encoding time and intra-buffer overhead
- Does NOT reduce the submit-wait overhead

---

## 6. NEURAL ENGINE (ANE) OFFLOAD FOR ATTENTION (Rank: LOW-MEDIUM)

### The Idea
Use Apple's Neural Engine for the attention computation (which is mostly matmuls) while the GPU handles MoE expert FFNs. Run them in parallel.

### First-Principles Reasoning
The M4 has a 38 TOPS Neural Engine that sits idle during llama.cpp inference. If we could offload attention to the ANE:
- GPU handles: expert routing + 8 expert FFN matmuls
- ANE handles: Q/K/V projections, attention scores, output projection
- They run in parallel, reducing wall-clock time

### Feasibility: VERY LOW
- The ANE is accessed through CoreML, not Metal
- CoreML requires ANE-compatible model formats (ML Program / MIL)
- Quantized formats (IQ3_S) are not supported by ANE -- it only does int8/fp16
- The latency to submit work to ANE and get results back might exceed the savings
- No way to synchronize Metal and ANE within a single pipeline

### Expected Impact: Theoretical 2x if both units are fully utilized
- But the practical overhead of CoreML submission + format conversion would likely negate any gains

---

## 7. INDIRECT COMMAND BUFFERS (ICBs) FOR GPU-AUTONOMOUS DECODE (Rank: MEDIUM)

### The Idea
Use Metal's Indirect Command Buffers to let the GPU encode its OWN next command buffer based on the current token's results. The GPU becomes self-scheduling.

### First-Principles Reasoning
Metal 2 introduced `MTLIndirectCommandBuffer` which lets compute/render commands be populated by GPU kernels at runtime. If the GPU can write its own commands:
1. GPU finishes token N's forward pass
2. GPU runs argmax on logits
3. GPU encodes the commands for token N+1's forward pass into an ICB
4. GPU executes the ICB immediately

No CPU round-trip needed.

### Feasibility: LOW
- Metal ICBs support `dispatchThreads` and `drawPrimitives` but with significant limitations
- ICBs cannot set pipeline states dynamically from GPU (the pipelines must be pre-set)
- ICBs cannot do `dispatchThreadgroups` with GPU-computed grid sizes
- The forward pass requires different pipeline states for different operations
- Maximum 16,384 commands per ICB (might not be enough for a full forward pass)

### Expected Impact: Could eliminate submission overhead entirely
- But the API limitations make it impractical for a full transformer forward pass

---

## 8. BATCH MULTIPLE SPECULATIVE TOKENS (Rank: MEDIUM-HIGH)

### The Idea
Use the model itself to predict K tokens greedily in a SINGLE command buffer, then verify them all at once. This is different from classic speculative decoding (which uses a draft model) -- we use the SAME model but trade compute for latency.

### How It Works
1. Run one forward pass to get token T1
2. On GPU (same command buffer): argmax -> embedding lookup -> run forward pass again for T2
3. Repeat for T3, T4, ... TK
4. Return all K tokens to CPU
5. CPU verifies (optional -- for greedy decoding, verification is not needed since we used the real model)

This is essentially Approach #1 (Speculative Multi-Token Command Buffer) stated differently, but with the important insight that for GREEDY decoding, there's no speculation -- every token is exact.

### For Non-Greedy Decoding
Pre-compute the cumulative distribution on GPU, draw a uniform random number, and binary-search the CDF. This is standard GPU sampling and is well-understood.

### Feasibility: HIGH (same as #1)
### Expected Impact: LINEAR with K (same as #1)

---

## Summary Ranking

| Rank | Approach | Expected Impact | Feasibility | Priority |
|------|----------|-----------------|-------------|----------|
| 1 | Multi-token command buffer | 3-8x | HIGH | BUILD THIS |
| 2 | Double-buffered pipeline | 30-40% | HIGH | Stack with #1 |
| 3 | Pre-fault + wire pages | 15-40% | MEDIUM | Quick experiment |
| 4 | Persistent GPU kernel | 10x | LOW | Research only |
| 5 | Fused mega-kernel | 10-20% | LOW-MEDIUM | Diminishing returns |
| 6 | Indirect command buffers | Eliminates overhead | LOW | API too limited |
| 7 | ANE offload | 2x theoretical | VERY LOW | Not practical |

**The clear winner is #1: Multi-Token Command Buffer.** It's the only approach that addresses the root cause (per-submission overhead) with a multiplicative improvement AND is feasible with existing Metal APIs. The others are either too limited in impact or too risky in feasibility.

The optimal strategy is: #1 (multi-token) + #3 (pre-fault pages) + #2 (pipelining). These three compose:
- Multi-token eliminates K-1 out of K submissions
- Pre-faulting reduces the remaining submission's page validation cost
- Pipelining overlaps CPU encoding with GPU compute

Combined estimate: 4 tokens per buffer * 1.3x from pre-fault * 1.3x from pipelining = ~6.8x improvement = ~190 tok/s theoretical.
