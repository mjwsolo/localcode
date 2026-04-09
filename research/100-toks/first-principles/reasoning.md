# First-Principles Analysis: Why Metal Takes 35ms Per Token

## The Observed Anomaly

- Prompt eval (batched): 318 tok/s = ~3.1ms per token
- Single-token decode: 28 tok/s = ~35.7ms per token
- The model is identical. The compute per token is identical. The ONLY difference is how work is submitted to Metal.

During prompt eval, hundreds of tokens are packed into ONE command buffer commit. During decode, each token requires its own commit-wait-commit cycle. This means the 35ms is NOT compute -- it's the overhead of the submit/wait boundary.

## Decomposing the 35ms: What MUST Happen

Let's trace what happens between `[cmd_buf commit]` and `[cmd_buf_last waitUntilCompleted]` returning:

### 1. Command Buffer Compilation (~2-5ms estimated)

When you call `[cmd_buf commit]`, Metal does NOT just hand raw dispatches to the GPU. It must:
- Validate all encoder state (pipelines, buffers, offsets)
- Resolve argument buffer references
- Compile the command buffer into GPU-native command stream
- This is NOT shader compilation (PSOs are pre-compiled), but it IS command stream linearization

For a Gemma 4 decode step with 30 layers, each layer having attention (Q/K/V projections, softmax, output projection) plus MoE routing (gate network, top-8 expert selection, 8 expert FFNs, combine), we're looking at roughly 300-500 compute dispatches per command buffer. Encoding these is work.

**Key insight from the code**: llama.cpp uses `n_cb=1` (one extra thread). The main thread encodes `n_nodes_0 = MAX(64, 0.1*n_nodes)` nodes into `cmd_bufs[n_cb]` and commits immediately. Then 1 extra thread encodes the rest into `cmd_bufs[0]`. For single-token decode, the graph might be ~400-600 nodes. So the main thread encodes ~64 nodes into one command buffer and commits, then the extra thread encodes ~340-540 into another.

That means we have 2 command buffers per decode step. Each needs compilation.

### 2. GPU Page Table / IOMapper Setup (~5-15ms estimated -- THIS IS THE BIG ONE)

The model weights are in a 10.4GB GGUF file accessed via mmap. Metal wraps this with `newBufferWithBytesNoCopy` using `MTLResourceStorageModeShared`. But here's the critical issue:

**mmap does not mean the pages are resident in physical memory.** The first time (or after memory pressure) a page is accessed, it triggers a page fault. For GPU access, this is MUCH worse than CPU page faults because:

- The GPU's IOMapper (IOMMU) needs valid page table entries for every page the command buffer might touch
- Metal must walk the buffer's virtual address range and ensure all pages are mapped in the GPU's address space
- For a 10.4GB mmap'd buffer, that's 2.5 MILLION 4KB pages (or ~640K 16KB pages on Apple Silicon)
- Even with residency sets (which llama.cpp uses), the system must validate that pages haven't been evicted since the last check

**The residency set mechanism is telling**: llama.cpp has `ggml_metal_device_rsets_keep_alive()` called at the start of every `graph_compute`. This requests that Metal keep the buffers resident. But requesting residency doesn't guarantee it -- the OS can still evict under memory pressure, and on a 16GB machine running a 10.4GB model, there IS memory pressure.

**Why batched is faster**: During prompt eval, you pay this page table validation cost ONCE for 512 tokens. During decode, you pay it ONCE for 1 token. Same fixed cost, amortized differently.

### 3. Inter-Process GPU Scheduling (~3-8ms estimated)

Apple Silicon has ONE GPU shared by:
- WindowServer (compositing every frame)
- Any other Metal-using apps
- The system compositor
- Our llama-server process

Metal's command queue is NOT a direct pipe to the GPU. It goes through:
1. `[cmd_buf commit]` -> Metal driver's command queue
2. Metal driver -> IOKit / AGX firmware submission
3. AGX firmware -> GPU hardware scheduler
4. GPU executes
5. GPU signals completion -> AGX firmware interrupt
6. Interrupt handler -> wakes waiting thread

Steps 2-3 and 5-6 involve kernel transitions (user -> kernel -> firmware -> hardware and back). Each transition has latency from:
- System call overhead (~1us each, negligible)
- Interrupt routing (~5-50us)
- Context switching if WindowServer has a frame pending (~1-5ms)
- **AGX firmware scheduler quantum**: The AGX firmware runs its own scheduler. It may not schedule our command buffer immediately if another context is active.

### 4. The `waitUntilCompleted` Wakeup Latency (~1-5ms)

When `[cmd_buf waitUntilCompleted]` is called, the calling thread blocks. When the GPU finishes:
1. GPU writes a completion fence
2. AGX firmware notices (via interrupt or polling)
3. Firmware signals the Metal driver
4. Driver wakes the blocked thread via `mach_msg` or similar
5. Thread gets scheduled by Darwin scheduler

This wakeup path has inherent latency. The thread may not be immediately scheduled if CPU cores are busy. On M4 with 10 cores, this should be fast, but the kernel-to-userspace transition still costs ~50-200us minimum.

### 5. Memory Coherency / Cache Management (~1-3ms estimated)

Apple Silicon uses a unified memory architecture, but "unified" doesn't mean "free coherence." When the GPU writes results (logits, KV cache updates), the CPU needs to see them. With `StorageModeShared`:
- The GPU may have written to its L2 cache but not flushed to system memory
- Metal must issue cache maintenance operations (clean + invalidate)
- For the KV cache write-back after each token, this could be significant

### 6. Autorelease Pool and Objective-C Overhead (~0.5-1ms)

Every `graph_compute` call creates an `@autoreleasepool` and does:
- `[queue commandBufferWithUnretainedReferences]` (allocates a command buffer from the pool)
- Multiple `[cmd_buf retain]` / `[cmd_buf release]` cycles
- `dispatch_apply` for parallel encoding
- Block creation/copying for `encode_async`

This is pure CPU overhead but adds up.

## The Synthesis: Where the 35ms Actually Goes

```
Command encoding (CPU):           3-5ms   (300+ dispatches to encode)
Page table validation/IOMapper:   8-15ms  (10.4GB mmap, 2.5M pages)
GPU scheduling + firmware:        5-8ms   (commit -> GPU start)
Actual GPU compute:               2-5ms   (memory-bound matmuls)
Completion signaling:             2-4ms   (GPU done -> thread wakes)
Cache coherency:                  1-2ms   (shared memory flush)
ObjC / autorelease overhead:      0.5-1ms
                                  --------
Total:                            ~21-40ms
```

## Why Prompt Eval Doesn't Suffer

With a batch of 512 tokens:
- Same encoding cost (~5ms) but amortized over 512 tokens = 0.01ms/tok
- Same page table validation (~12ms) but amortized = 0.02ms/tok
- Same scheduling overhead (~6ms) amortized = 0.01ms/tok
- GPU compute scales linearly with batch (but is efficiently parallelized)
- Total overhead per token: ~0.05ms + actual compute
- This explains 318 tok/s perfectly: almost all time is useful compute

## The Core Insight

**The 35ms is dominated by FIXED PER-SUBMISSION costs that don't scale with the amount of work.** The GPU finishes in microseconds-to-low-milliseconds, but the entire submit-wait-return pipeline has ~30ms of ceremony around it.

The path to 100 tok/s (10ms/token) requires eliminating the per-token submission overhead entirely. You cannot optimize 35ms of fixed overhead down to 10ms -- you must architecturally eliminate the submit-wait cycle from the critical path.
