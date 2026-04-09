# Metal GPU Sleep/Wake Overhead

**Sources**:
- https://developer.apple.com/forums/thread/46817
- https://developer.apple.com/videos/play/tech-talks/10580/
- https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html

## The GPU Sleep Problem

When the GPU finishes a command buffer and no new work is queued, it enters a low-power
sleep state. Waking from sleep has significant latency.

### Measured Impact

- **2-4x performance loss** measured by Apple engineers, even on "extremely large
  machine learning workloads"
- **2.5ms baseline** submission overhead on older hardware (forum: "Metal Compute
  Never beats 2.5ms?")
- Apple recommends **30ms of GPU work queued at any time** to maintain optimal
  performance/power states

### Why 35ms?

The 35ms observed in our decode is likely a combination of:

1. **GPU power state ramp-down + ramp-up** (~5-15ms): After the ~1ms of actual compute
   for a single-token decode, the GPU clock drops. Next command buffer must wait for
   clock ramp-up.

2. **Command buffer scheduling overhead** (~2-5ms): JIT compilation (first invocation),
   resource validation, memory wiring checks for mmap'd buffers.

3. **mmap page fault servicing** (~10-25ms): The 10.4GB model is memory-mapped. Each
   decode touches different expert weights (MoE). If expert pages were evicted under
   memory pressure, the GPU must wait for page-in from SSD. Even without eviction,
   TLB refills for scattered expert access across the 10.4GB address range add latency.

4. **CPU thread scheduling** (~1-5ms): `waitUntilCompleted` blocks the calling thread.
   When the GPU signals completion, the CPU thread must be rescheduled by the OS.
   Under load, this adds jitter.

### The Clock Scaling Evidence

The fact that prompt eval achieves 318 tok/s (3.1ms/token amortized over hundreds of
tokens per command buffer) while decode gets 28 tok/s (35ms/token with 1 token per
command buffer) strongly suggests the GPU is entering a low-power state between
single-token command buffers.

At 318 tok/s, the GPU is running at full clock speed with sustained work. At 28 tok/s,
each command buffer has ~1ms of actual compute followed by a gap where the GPU idles.

## Solution

Never let the GPU go idle. Use one of:
1. MTLSharedEvent double-buffering (proven <50us overhead)
2. Pre-queue next command buffer before waiting for current one
3. Keep-alive heartbeat command buffers (crude but effective)
