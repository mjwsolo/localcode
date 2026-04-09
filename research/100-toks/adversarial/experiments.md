# Experimental Protocol: Attacking the 35ms Claim

## Overview

Each experiment tests ONE variable while holding everything else constant.
All experiments run from `benchmark.py` which outputs structured results.

---

## Experiment 1: Baseline — Trivial Kernel Commit Latency

**Hypothesis**: A trivial kernel (1 threadgroup, 1 thread, writes a single
float) has commit-to-completion latency under 3ms. If true, the 35ms is
NOT a Metal driver floor.

**Protocol**:
1. Create a trivial compute kernel: `output[0] = 1.0`
2. Create a small MTLBuffer (4 bytes)
3. Commit, measure wall-clock time from commit to waitUntilCompleted
4. Repeat 100 times, discard first 10 (warm-up)
5. Report: min, median, p95, max

**If result < 5ms**: The 35ms is model-specific, not a Metal floor.
**If result ~ 35ms**: The overhead is truly in the Metal driver regardless of workload.

---

## Experiment 2: Timestamp Decomposition

**Hypothesis**: The 35ms can be decomposed into pre-GPU (scheduling) and
post-GPU (signal/sync) components using Metal's built-in timestamps.

**Protocol**:
1. Run the trivial kernel from Experiment 1
2. After completion, read:
   - `commandBuffer.kernelStartTime` (when GPU started)
   - `commandBuffer.kernelEndTime` (when GPU finished)
   - Wall-clock time of `commit` call
   - Wall-clock time of `waitUntilCompleted` return
3. Compute:
   - Pre-GPU delay = kernelStartTime - commit_time
   - GPU time = kernelEndTime - kernelStartTime
   - Post-GPU delay = wait_return_time - kernelEndTime

**Expected outcome**: The pre-GPU delay dominates (scheduling overhead).

---

## Experiment 3: MTLBuffer vs mmap

**Hypothesis**: mmap'd memory causes additional page table validation
per commit. GPU-private MTLBuffer eliminates this overhead.

**Protocol**:
1. Allocate MTLBuffer(s) of increasing sizes: 1MB, 100MB, 1GB, 4GB, 8GB
2. For each size:
   a. Fill buffer with random data
   b. Run a trivial kernel that reads from the buffer (sum reduction)
   c. Measure commit-to-completion latency (100 iterations)
3. Repeat with mmap'd files of the same sizes:
   a. Create a temp file, mmap it
   b. Wrap in MTLBuffer using `newBufferWithBytesNoCopy` or use
      `useResource` on the compute encoder
   c. Measure commit-to-completion latency

**Key comparison**: Does mmap add measurable overhead vs resident MTLBuffer?

---

## Experiment 4: Scaling with Buffer Size

**Hypothesis**: Commit latency scales with the total size of resources
bound to the command buffer.

**Protocol**:
1. Allocate MTLBuffers of sizes: 4B, 1KB, 1MB, 100MB, 1GB, 4GB, 8GB
2. Bind ALL buffers to a single compute encoder
3. Kernel only reads from the 4B buffer (trivial work)
4. Measure commit-to-completion time

**Key insight**: If latency is constant regardless of total bound resource
size, then resource validation is NOT the bottleneck.

---

## Experiment 5: GPU Pre-Warming (Power State)

**Hypothesis**: GPU power state transitions account for 10-20ms of the
35ms. Keeping the GPU warm eliminates this.

**Protocol**:
1. **Cold GPU**: Wait 1 second (no GPU activity), then commit
2. **Warm GPU**: Run a continuous heartbeat (commit a dummy kernel every
   2ms on a separate queue), then commit our test kernel
3. Compare latencies

**If warm GPU shows 15-20ms less**: Power cycling is the dominant factor
and we can fix it with a heartbeat thread.

---

## Experiment 6: Scaling with Dispatch Count

**Hypothesis**: Commit latency scales with the number of compute dispatches
encoded in the command buffer.

**Protocol**:
1. Encode 1, 10, 100, 500, 1000, 2500 dispatches of the trivial kernel
   into a single command buffer
2. Measure commit-to-completion time for each count
3. Plot dispatch_count vs latency

**If linear**: Kernel fusion (reducing from 2677 to ~100 dispatches) would
cut scheduling overhead proportionally.
**If constant**: Dispatch count is not a factor.

---

## Experiment 7: Command Buffer Options

**Hypothesis**: Different command buffer creation options affect overhead.

**Protocol**:
Test each combination:
1. `commandBuffer` (default, retained references)
2. `commandBufferWithUnretainedReferences`
3. `MTLCommandBufferDescriptor` with `.errorOptions = .none`
4. Different command queue priorities (if available)

Measure commit-to-completion for the trivial kernel with each option.

---

## Experiment 8: Pipelined Double Buffering

**Hypothesis**: Back-to-back commits without waiting reduce per-token
overhead by eliminating the scheduling gap.

**Protocol**:
1. **Serial**: Commit A, wait, commit B, wait (measure total)
2. **Pipelined**: Commit A, encode B, commit B, wait B (measure total)
3. **Rapid fire**: Commit 10 buffers without waiting, wait on last only

If pipelined total < 2x serial single, overlap is working and the
scheduling gap is being hidden.

---

## Experiment 9: MTLSharedEvent vs waitUntilCompleted

**Hypothesis**: `waitUntilCompleted` has thread-wakeup overhead that
MTLSharedEvent-based signaling avoids.

**Protocol**:
1. **waitUntilCompleted**: Standard blocking wait
2. **addCompletedHandler**: Async callback, measure in callback
3. **MTLSharedEvent**: Signal a shared event, spin-wait on its value
4. **Spin on status**: Poll `commandBuffer.status` in a tight loop

Compare the wall-clock time each method reports.

---

## Expected Outcomes Matrix

| Experiment | If 35ms is a driver floor | If 35ms is reducible |
|-----------|--------------------------|---------------------|
| 1. Trivial kernel | ~35ms | <5ms |
| 2. Timestamps | Uniform 35ms pre-GPU | Most time in specific phase |
| 3. MTLBuffer vs mmap | Same latency | mmap significantly worse |
| 4. Buffer size scaling | Constant | Linear or sublinear |
| 5. GPU warming | No change | 15-20ms reduction |
| 6. Dispatch count | Constant | Linear scaling |
| 7. Buffer options | No change | Some options faster |
| 8. Pipelining | No overlap | Significant overlap |
| 9. Event signaling | Same as wait | 1-5ms reduction |

If ANY of experiments 1, 3, 5, or 6 show the "reducible" outcome,
the 35ms claim is broken and we have a specific attack vector.
