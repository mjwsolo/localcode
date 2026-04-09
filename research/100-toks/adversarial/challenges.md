# Adversarial Challenges to the "35ms Metal Scheduling Floor" Claim

## The Claim Under Attack

> "Metal command buffer scheduling takes 35ms per commit on M4, and this cannot
> be reduced. The GPU compute is 0ms. This is a driver-level floor."

## Evidence Used to Support the Claim
1. `waitUntilCompleted` shows 35ms per token
2. Spin-wait polling shows the same 35ms
3. `GPUStartTime` to `GPUEndTime` is ~0ms
4. Prompt eval amortizes the 35ms across a batch

## Challenges

### Challenge 1: The 35ms Was Never Measured in Isolation

The 35ms was measured **during inference** with:
- A 10.4GB mmap'd model loaded
- WindowServer and system compositor running
- The llama-server HTTP stack active
- KV cache writes happening
- CPU sampling running between tokens

No one has ever measured a BARE Metal command buffer commit-to-completion
on this specific M4 with nothing else running. The Apple Dev Forums cite
a **2.5ms** floor for trivial workloads. If our trivial kernel benchmarks
at 2.5ms instead of 35ms, the remaining 32.5ms is NOT "Metal scheduling" --
it is something we are doing wrong.

**Experiment**: Run `benchmark.py` Test 1 (trivial kernel, no model loaded).

---

### Challenge 2: GPU Power State Cycling Is Not Inevitable

The existing research (paper 01) acknowledges GPU power state transitions
could account for 15-20ms. But this was stated as fact, never tested.

If we keep the GPU warm with a continuous background heartbeat kernel
(a dummy dispatch every 5ms on a separate command queue), the GPU may
never enter low-power state. If the 35ms drops to 15ms, we just found
20ms of recoverable overhead.

**Experiment**: Run `benchmark.py` Test 5 (GPU pre-warming).

---

### Challenge 3: mmap Page Table Validation Was Assumed, Not Measured

The claim is that Metal must validate page tables for 10.4GB of mmap'd
data on every commit. But:

1. Metal uses **residency sets** (`MTLResidencySet`) to declare which
   resources are needed. If resources are already resident, validation
   should be fast.
2. llama.cpp's `ggml_metal_device_rsets_keep_alive()` keeps residency
   sets alive, but does it actually call `requestResidency()` on them?
3. What if we allocate a GPU-private `MTLBuffer` and copy the model into
   it instead of mmap? This eliminates ALL page table uncertainty because
   GPU-private buffers are guaranteed resident.

If MTLBuffer-resident weights show <5ms commit overhead, mmap is the
problem and the 35ms is NOT a Metal driver floor.

**Experiment**: Run `benchmark.py` Test 3 (MTLBuffer vs mmap comparison).

---

### Challenge 4: Resource Set Size Was Never Varied

The 35ms was measured with ONE model size (10.4GB). What if:
- A 100MB model shows 3ms? (overhead scales with resource size)
- A 1GB model shows 10ms? (overhead is O(n) in buffer count or size)
- A 100MB model ALSO shows 35ms? (overhead is truly fixed)

If overhead scales with resource size, we can reduce it by:
- Splitting the model into smaller MTLBuffers
- Only binding the buffers needed per token (active experts + attention)
- Using argument buffers to avoid per-commit resource binding

**Experiment**: Run `benchmark.py` Test 4 (scaling with buffer size).

---

### Challenge 5: The Number of Encoded Dispatches Matters

Our decode graph has ~2677 nodes encoded into the command buffer.
Each dispatch requires Metal to:
- Validate the pipeline state
- Set up threadgroup memory
- Configure the dispatch grid

What if a command buffer with 10 dispatches commits in 2ms but one
with 2677 dispatches takes 35ms? That would mean the overhead is
O(n_dispatches), and kernel fusion (reducing dispatch count) is the
correct fix -- not "it's a driver floor."

**Experiment**: Run `benchmark.py` Test 6 (scaling with dispatch count).

---

### Challenge 6: The Measurement Methodology May Be Wrong

`GPUStartTime` and `GPUEndTime` on the command buffer measure when the
GPU **starts and finishes executing kernels**. They do NOT measure:
- Time between `commit` and `GPUStartTime` (scheduling delay)
- Time between `GPUEndTime` and `waitUntilCompleted` returning (signal delay)

If the 35ms is mostly in the commit-to-GPUStart gap, it is scheduling
overhead. If it is in GPUEnd-to-wait-return, it is signal/synchronization
overhead. These have different fixes.

**Experiment**: Run `benchmark.py` Test 2 (timestamp decomposition).

---

### Challenge 7: `commandBufferWithUnretainedReferences` May Be Faster

llama.cpp already uses `commandBufferWithUnretainedReferences` which
skips reference counting on resources. But are there other command buffer
options that reduce overhead?

- `MTLCommandBufferDescriptor` with `.retainedReferences = false`
- `.errorOptions = .none` (skip error checking)
- Using a shared command queue vs dedicated compute queue
- Priority hints on the command queue

**Experiment**: Run `benchmark.py` Test 7 (command buffer options).

---

### Challenge 8: Double Buffering / Pipelining Is Untested

Instead of commit-wait-commit-wait, what about:
1. Commit buffer A
2. While A executes, encode buffer B
3. Commit buffer B immediately after A completes (no scheduling gap)

If the GPU has zero idle time between buffers, the scheduling overhead
overlaps with the next encode. llama.cpp partially does this with async
dispatch, but the synchronize() call between tokens serializes everything.

**Experiment**: Run `benchmark.py` Test 8 (pipelined double buffering).

---

### Challenge 9: The 2.5ms Forum Floor Suggests 14x Overhead Is Model-Specific

Apple's own Dev Forums document a ~2.5ms floor. We see 35ms -- that is
14x higher. The question is whether this 14x comes from:
- Resource set size (mmap'd 10.4GB)
- Dispatch count (2677 nodes)
- Page table pressure (TLB misses)
- GPU power cycling
- Some combination

If we can identify which factor contributes what fraction, we can attack
the dominant one. If power cycling is 20ms and mmap validation is 13ms,
fixing EITHER ONE drops us to 15-22ms = 45-66 tok/s.

---

### Challenge 10: Nobody Tried `MTLEvent` Signaling Instead of `waitUntilCompleted`

`waitUntilCompleted` is a coarse synchronization primitive. Metal provides
finer-grained synchronization via:
- `MTLEvent` / `MTLSharedEvent` -- signal/wait between command buffers
- `addCompletedHandler` -- async callback instead of blocking wait
- `encodeSignalEvent` / `encodeWaitForEvent` within command buffers

If the 35ms includes thread-wakeup latency from `waitUntilCompleted` (the
calling thread sleeps and must be rescheduled by the OS), then using an
event-based callback with a spin-waiting thread could shave 1-5ms.

The existing spin-wait test showed "same 35ms" but was it spinning on
`status` polling? The polling interval and precision matter.

**Experiment**: Run `benchmark.py` Test 9 (MTLSharedEvent signaling).

---

## Summary: The 35ms Is a Composite, Not a Monolith

The existing research treats 35ms as a single number. In reality it is
likely composed of:

| Component | Estimated Range | Reducible? |
|-----------|-----------------|------------|
| GPU power state ramp-up | 10-20ms | YES (keep GPU warm) |
| mmap page table validation | 5-15ms | YES (MTLBuffer, smaller resource sets) |
| Command buffer encoding/JIT | 2-5ms | MAYBE (fewer dispatches) |
| Scheduling quantum | 2-5ms | MAYBE (pipelining) |
| Thread wakeup / signal | 1-3ms | YES (event-based sync) |

If all reducible components are addressed, the floor could be as low as
2-5ms per token = 200-500 tok/s theoretical maximum.

Even conservatively, reducing from 35ms to 15ms = 66 tok/s, a 2.4x win.
