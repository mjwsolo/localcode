# Anukari: Achieving <50us Metal Compute Overhead

**Source**: https://anukari.com/blog/devlog/huge-macos-performance-improvements
**Context**: Real-time audio synthesis using Metal compute kernels

## Problem (identical to ours)

Anukari needed to launch Metal compute kernels with minimal latency between dispatches.
Standard commit-wait-commit pattern had unacceptable overhead.

## Solution: MTLSharedEvent Double-Buffering

An Apple engineer recommended the following technique:

1. Create TWO command buffers (A and B)
2. Command buffer B includes `encodeWait` for an MTLSharedEvent before its first kernel
3. Command buffer B includes `encodeSignal` for a different MTLSharedEvent after its last kernel
4. While A is executing on GPU:
   - CPU encodes the NEXT command buffer
   - CPU writes dynamic parameters to device memory
   - CPU signals MTLSharedEvent to unblock B's kernels
5. CPU waits for B's completion event (or polls via MTLSharedEvent listener)

### Key Details

- Parameters that aren't known until the next iteration can't be kernel parameters --
  they must be written to device memory AFTER commit but BEFORE the event signal
- The encoding work (~50us) is fully overlapped with GPU execution
- Result: **<50us scheduling/waiting overhead** (down from milliseconds)

## Critical Measurement

> "The encoding work takes around 50us and all of that can be saved by doing it
> in parallel with the GPU work."

## Why This Matters for Us

Our current flow:
```
[GPU: compute token N] -> [CPU: waitUntilCompleted] -> [CPU: sample] -> [CPU: encode N+1] -> [GPU: commit+execute N+1]
                                                                         ^--- 35ms gap ---^
```

With MTLSharedEvent double-buffering:
```
[GPU: compute N] [GPU: wait_event -> compute N+1] [GPU: wait_event -> compute N+2]
     |                    |                              |
[CPU: encode N+1, signal] [CPU: sample N, encode N+2, signal]
```

The GPU never goes idle. The 35ms overhead becomes ~50us.
