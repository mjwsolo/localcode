# Prioritized Solutions: Metal Decode Latency

Solutions ranked by (Impact x Feasibility). Target: reduce 35ms/token to <10ms/token.

## Solution 1: MTLSharedEvent Double-Buffered Command Pipeline
**Impact: EXTREME (35ms -> <2ms overhead)**
**Feasibility: MEDIUM (requires Metal backend changes in llama.cpp)**
**Priority: #1**

### What

Replace the synchronous commit-wait-commit pattern with a pipelined pattern using
MTLSharedEvent for coordination. The GPU never goes idle between tokens.

### How It Works

```
Current (serial):
  [GPU exec N] --gap:35ms-- [GPU exec N+1] --gap:35ms-- [GPU exec N+2]
  
Proposed (pipelined):
  [GPU: wait_eventA -> exec N -> signal_eventB]
  [GPU: wait_eventC -> exec N+1 -> signal_eventD]  (committed while N runs)
  [CPU: encode N+1, write params, signal eventC]    (overlapped with GPU N)
```

### Why It Works

Anukari (real-time audio on Metal) proved this achieves **<50us** scheduling overhead,
down from milliseconds. An Apple engineer specifically recommended this pattern.

The key insight: the GPU has a built-in event wait mechanism. A command buffer can be
committed with a "wait for event X" at the start. The GPU doesn't start executing
until the CPU signals event X. This means:

1. CPU commits command buffer N+1 immediately after N (GPU doesn't start yet)
2. CPU waits for N's completion event
3. CPU reads N's output, computes argmax, writes N+1's input token to device memory
4. CPU signals event X to unblock N+1 on the GPU
5. GPU starts N+1 immediately -- zero idle time

### Code Changes Required

File: `ggml/src/ggml-metal/ggml-metal-context.m`

1. Add `MTLSharedEvent` objects to `struct ggml_metal`
2. In `ggml_metal_graph_compute()`:
   - Before encoding, insert `encodeWaitForEvent` on the command buffer
   - After encoding, insert `encodeSignalEvent`
   - Commit immediately (don't wait)
3. New function `ggml_metal_signal_start(ctx)`:
   - Called by the CPU after it has written the input token
   - Signals the MTLSharedEvent to unblock the GPU
4. Replace `waitUntilCompleted` in `synchronize()` with event-based wait

File: `ggml/src/ggml-backend.cpp`

5. Change `ggml_backend_graph_compute()` to use the pipelined path:
   - Call `graph_compute_async()` (encodes + commits with event wait)
   - Don't call `synchronize()` immediately
   - Instead, return a "future" or flag that the result is pending
   - The caller (decode loop in llama.cpp) can prepare the next token
     while waiting for the GPU

### Risk

- Requires careful double-buffering of input/output tensors
- The "signal after sampling" path must be correct or GPU hangs
- Testing: compare output tokens between serial and pipelined modes


## Solution 2: On-GPU Argmax (Eliminate CPU Roundtrip for Greedy Decode)
**Impact: HIGH (eliminates ~1-5ms CPU roundtrip per token)**
**Feasibility: MEDIUM (custom Metal kernel)**
**Priority: #2**

### What

Compute argmax of logits on the GPU. Write the result token ID to a small Metal buffer.
The next decode step reads the token ID directly from GPU memory -- no CPU involvement
for greedy sampling.

### How It Works

For greedy decoding (temperature=0, top_k=1):
1. Final matmul produces logits tensor on GPU (vocab_size = 262144 for Gemma 4)
2. Custom Metal kernel computes `argmax(logits)` -> single int32
3. Next decode step reads token ID from same Metal buffer
4. CPU only needs to read the token ID for display (async, non-blocking)

For non-greedy sampling (temperature>0):
- More complex: need GPU-side softmax + multinomial sampling
- Could use Metal's `simd_shuffle` for parallel reduction
- Random number generation via GPU-side PRNG (Metal supports this)

### Code Changes

1. New Metal kernel `kernel_argmax` in `ggml-metal.metal`:
```metal
kernel void kernel_argmax(
    device const float * logits [[buffer(0)]],
    device int32_t * result [[buffer(1)]],
    constant int32_t & n_vocab [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
    
    // Parallel reduction: each thread finds max in its chunk
    // Then simd_shuffle + threadgroup reduction to find global max
    threadgroup float shared_max[32];
    threadgroup int32_t shared_idx[32];
    
    float local_max = -INFINITY;
    int32_t local_idx = 0;
    
    for (int i = tid; i < n_vocab; i += 1024) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }
    
    // SIMD reduction within thread group
    // ... (standard parallel reduction pattern)
}
```

2. Chain this kernel in the same command buffer as the forward pass
3. Read result from Metal buffer (unified memory -- just a pointer read)

### Synergy with Solution 1

Combined with double-buffering, the GPU can:
1. Compute forward pass for token N
2. Compute argmax for token N (same command buffer)
3. Write token N result to shared buffer
4. Signal event -> CPU reads result for display
5. Next command buffer reads token N result and starts token N+1

This eliminates the CPU from the critical path entirely for greedy decode.

### Risk

- Argmax kernel must handle vocab_size=262144 efficiently
- Non-greedy sampling on GPU is more complex (but greedy covers most coding use cases)
- Need fallback path for when CPU needs to do complex sampling


## Solution 3: Pre-Queue Next Command Buffer
**Impact: HIGH (prevents GPU sleep)**
**Feasibility: HIGH (small code change)**
**Priority: #3 (quick win)**

### What

Before calling `waitUntilCompleted` on command buffer N, encode and commit
command buffer N+1 with a placeholder input. Then wait for N. When N completes,
patch N+1's input and let it execute.

This is simpler than Solution 1 but less optimal -- the GPU still idles briefly
while the CPU patches the input. However, it prevents the GPU from entering
deep sleep because work is already queued.

### How It Works

```
1. Commit command buffer N
2. Encode command buffer N+1 (with dummy input token)
3. Commit N+1 (GPU queues it but N must finish first)
4. waitUntilCompleted(N)
5. CPU computes argmax, patches N+1's input token in unified memory
   (risky: N+1 may have already started with wrong input)
```

### Problem

This approach has a race condition: N+1 might start executing before the CPU
patches the input token. This is why Solution 1 (MTLSharedEvent) is superior --
the event mechanism provides a clean synchronization point.

### Alternative: Keep-Alive Heartbeat

Submit tiny no-op command buffers between decode steps to keep the GPU clock up:
```
[GPU: decode N] [GPU: noop] [GPU: noop] [GPU: decode N+1]
```

This is crude but might recover 5-15ms of the sleep/wake overhead with minimal
code changes.


## Solution 4: Reduce Graph Splits from 2 to 1
**Impact: MEDIUM (eliminates one command buffer boundary per token)**
**Feasibility: MEDIUM-HIGH (model loading change)**
**Priority: #4**

### What

Currently there are 2 graph splits per decode step. Each split is a separate
command buffer. Reducing to 1 split means 1 command buffer per token instead of 2.

### How

Graph splits happen because some tensors are on different backends (e.g., CPU vs GPU).
With `-ngl 999` and full GPU offload, splits likely occur because:
1. Input embedding lookup might be on CPU
2. Or the mmap buffer and compute buffer are treated as different backends

Investigate:
- Why exactly are there 2 splits?
- Can the split-causing operation be moved to GPU?
- `GGML_SCHED_DEBUG=2` will print split details

### Code Change

Run with `GGML_SCHED_DEBUG=2` environment variable and analyze which operations
cause the split. Then either:
- Move the operation to GPU backend
- Or merge the two backends into one allocation


## Solution 5: 2MB Superpages for mmap'd Model
**Impact: MEDIUM (reduces TLB overhead, estimated 5-15ms improvement)**
**Feasibility: LOW (macOS superpage support is limited)**
**Priority: #5**

### What

Replace standard 16KB mmap pages with 2MB superpages for the model file.
This reduces TLB entries needed from 650K to ~5,200.

### How

On macOS, superpages require:
1. `mach_vm_allocate` with `VM_FLAGS_SUPERPAGE_SIZE_2MB`
2. Contiguous physical memory (needs fresh reboot)
3. Copy model data into superpage-backed memory
4. Use this memory region instead of mmap for Metal buffer

### Why Low Feasibility

- macOS superpage support is poorly documented
- Requires contiguous physical memory
- Can't use mmap directly with superpages
- Would need a custom model loading path

### Alternative: Pre-fault All Pages

Instead of superpages, pre-read the entire model file into memory at startup:
```c
// After mmap, touch every page to force page-in
for (size_t i = 0; i < model_size; i += 16384) {
    volatile char x = ((char*)model_data)[i];
}
```

This doesn't help with TLB misses but eliminates page faults during decode.
The `rsets_keep_alive` mechanism in llama.cpp already does something similar
with residency sets, but it's worth verifying it's working.


## Solution 6: Fused Multi-Token Decode
**Impact: EXTREME (amortizes overhead across N tokens)**
**Feasibility: LOW (requires model architecture changes)**
**Priority: #6 (research)**

### What

Instead of decoding 1 token per forward pass, decode K tokens speculatively.
Even without a draft model, you can:
1. Run forward pass with beam width K
2. Take top-K tokens as candidates
3. Verify in one pass
4. Accept 1-K tokens per command buffer

### Why Low Feasibility

- Gemma 4 wasn't trained for multi-token prediction
- Increases compute by K for potentially <K accepted tokens
- Speculative decode was already tested and found slower (21.9 vs 27 tok/s)

### Alternative: Batch Multiple Independent Requests

If serving multiple requests, batch them into a single command buffer.
For single-user CLI, this doesn't apply.


## Impact Summary

| Solution | Current | Projected | Speedup | Effort |
|----------|---------|-----------|---------|--------|
| 1. MTLSharedEvent pipeline | 35ms | 5-8ms | 4-7x | 2-3 weeks |
| 2. On-GPU argmax | 35ms | 33ms* | 1.06x | 1 week |
| 3. Pre-queue / keep-alive | 35ms | 20-25ms | 1.4-1.8x | 2-3 days |
| 4. Reduce graph splits | 35ms | 25-30ms | 1.2-1.4x | 1 week |
| 5. 2MB superpages | 35ms | 25-30ms | 1.2-1.4x | 2 weeks |
| 6. Fused multi-token | 35ms | 10-15ms | 2-3x | months |

*Solution 2 alone saves little, but combined with Solution 1 it enables the GPU
to chain decode steps without ANY CPU involvement for greedy decode.

**Recommended attack order**: 3 (quick win) -> 4 (investigate) -> 1+2 (the real fix)

Combined Solutions 1+2+4 projected result: **5-8ms/token = 125-200 tok/s**
