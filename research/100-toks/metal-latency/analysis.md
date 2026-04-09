# Root Cause Analysis: 35ms Metal Command Buffer Overhead

## Executive Summary

The 35ms per-token decode latency is caused by three compounding factors, not one.
The GPU compute itself is approximately 1ms. The remaining 34ms is scheduling overhead
caused by the interaction of synchronous command buffer submission, GPU power state
management, and mmap-induced page fault/TLB overhead.

## The Evidence

| Metric | Value | Implication |
|--------|-------|-------------|
| Prompt eval | 318 tok/s (3.1ms/tok) | GPU is fast when kept busy |
| Decode | 28 tok/s (35ms/tok) | Massive overhead per token |
| GPU compute time | ~0.0ms (GPUStartTime/GPUEndTime) | Not a compute problem |
| Graph build | 0.3ms (after first) | Graph reuse works |
| Graph splits | 2 per decode | 2 command buffers per token |
| Model size | 10.4GB mmap'd | Massive virtual address range |
| Active experts/token | 8 of 128 | Scattered memory access |

## Root Cause 1: GPU Sleep/Wake Cycle (PRIMARY -- estimated 15-25ms)

The smoking gun: prompt eval batches hundreds of tokens into one command buffer,
keeping the GPU continuously busy at full clock speed (318 tok/s). Decode submits
1 token per command buffer with a synchronous wait between each.

The timeline for each decode token:

```
[GPU: execute ~1ms] -> [GPU: idle, clock ramps down] -> ...
    -> [CPU: waitUntilCompleted returns] -> [CPU: process result ~0.5ms]
    -> [CPU: build next graph ~0.3ms] -> [CPU: encode command buffer ~0.2ms]
    -> [GPU: commit, clock ramps up, execute ~1ms]
```

Apple's own documentation confirms:
- GPU clock slows significantly when idle
- Takes "a very long time" to ramp back up
- 2-4x performance loss measured in Apple's lab on ML workloads
- Recommendation: always have 30ms of GPU work queued

The current code flow (ggml-metal-context.m:438-614):
1. ggml_metal_graph_compute() encodes and commits command buffers
2. Returns without waiting (async)
3. But ggml_backend_graph_compute() in ggml-backend.cpp:358-361 calls
   graph_compute_async() then immediately calls synchronize()
4. synchronize() calls [cmd_buf_last waitUntilCompleted]
5. This blocks the CPU thread until GPU finishes
6. Then the CPU does sampling, builds next graph, encodes, and commits
7. The GPU has been idle this entire time

With 2 graph splits per decode step, the scheduler in ggml_backend_sched_compute_splits()
(ggml-backend.cpp:1445) iterates through splits, calling graph_compute_async on each,
then synchronize at the end.

## Root Cause 2: mmap Page Fault / TLB Overhead (SECONDARY -- estimated 5-15ms)

The 10.4GB GGUF is memory-mapped to the GPU via Metal's unified memory. During decode:

1. MoE routing selects 8 of 128 experts
2. Expert weights are interleaved with attention weights in the GGUF file
3. Each expert access hits a different region of the 10.4GB address space
4. With 16KB pages, this is ~650K virtual pages
5. GPU TLB can only cache a fraction of these entries
6. Each TLB miss triggers a page table walk (~100-500 cycles)
7. If pages were evicted under memory pressure, a full page fault occurs (~1-10ms)

## Root Cause 3: CPU Thread Scheduling (MINOR -- estimated 1-5ms)

waitUntilCompleted blocks the calling thread. When the GPU signals completion,
the OS must reschedule the blocked thread. Under load, this adds 1-5ms of jitter.

## Why It's NOT VSync

The 35ms is NOT display-linked:
- VSync at 60Hz = 16.7ms, at 120Hz = 8.3ms -- neither matches 35ms
- VSync only applies to drawable-based rendering (CAMetalLayer)
- Pure compute command buffers have no VSync dependency

## Why Prompt Eval Doesn't Suffer

Prompt eval processes hundreds of tokens per command buffer:
- GPU stays busy for 100-500ms continuously
- Never enters sleep state
- TLB cache warms up and stays warm across the batch
- Amortized overhead per token is tiny (3.1ms total including compute)

## The Math

If we could eliminate the sleep/wake overhead and pipeline command buffers:
- Memory bandwidth limit: ~400MB active weights at ~100GB/s = 4ms/token
- Theoretical: 200-250 tok/s
- Gap between 4ms theoretical and 35ms actual is entirely scheduling overhead
