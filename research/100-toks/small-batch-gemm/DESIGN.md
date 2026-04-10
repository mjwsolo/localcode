# Small-Batch MoE GEMM: Threshold Tuning for Speculative Decode

## Problem

llama.cpp's `MUL_MAT_ID` operation (used for MoE expert dispatch) has a threshold
at line 2420 of `ggml-metal-ops.cpp`:

```c
const int ne21_mm_id_min = is_tq_weight ? 1 : 32;
```

For non-TurboQuant weights (including IQ3_S), any batch size < 32 falls through to
the GEMV path (`kernel_mul_mv_id`), which dispatches one threadgroup per (expert, token)
pair. Each threadgroup independently reads the expert weight matrix from device memory.

With speculative decoding at batch=5, this means each expert's weights are read 5 times
from device memory -- once per speculated token. This is pure waste.

## How the Existing GEMM Kernel Tiles (kernel_mul_mm_id)

The GEMM path uses a two-phase approach:

### Phase 1: Map (kernel_mul_mm_id_map0)
- Runs one thread per expert (128 threads for Gemma 4)
- Each expert scans all tokens to find which ones route to it
- Builds a compact list: `ids_i32[expert][0..n_routed-1]` = token indices
- Stores count per expert in `tpe_u32[expert]`

### Phase 2: Multiply (kernel_mul_mm_id)
- Grid: `((ne21+31)/32, (ne01+63)/64, ne02)` = `(tokens_padded/32, rows/64, n_experts)`
- Each threadgroup handles a 64x32 output tile: 64 output rows x 32 tokens
- Key constants:
  - `NR0 = 64` (output rows per tile)
  - `NR1 = 32` (tokens per tile)
  - `NK = 32` (inner dimension chunk)

### Weight Sharing Analysis

The critical loop (line 11311):
```metal
for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
    // Load 64 rows x 32 columns of weights into threadgroup memory (sa)
    // Load 32 tokens x 32 columns of inputs into threadgroup memory (sb)
    // Compute 64x32 output tile via simdgroup matmul
}
```

**Weight data (sa)** is loaded from `src0` (expert weights) -- each thread loads its
assigned row. This data is stored in threadgroup shared memory (4096 bytes for sa).

**Input data (sb)** is loaded from `src1` -- each thread loads from its assigned token.
Also stored in threadgroup shared memory.

**The simdgroup_multiply_accumulate computes the full 64x32 = 2048-element output tile.**

This means: for a tile of 32 tokens, each weight element is read from device memory ONCE
and reused across all 32 tokens via threadgroup shared memory. The bandwidth saving is
(K-1)/K where K = number of tokens in the tile.

### What Happens at Batch=5?

With ne21=5 (5 tokens), the grid becomes `((5+31)/32, (704+63)/64, 128)` = `(1, 11, 128)`.

- Only 1 threadgroup column for the token dimension (since 5 < 32)
- `neh1` (tokens routed to this expert) will be ~5 (or fewer per expert)
- `nr1 = min(neh1 - 0, 32)` = 5

The kernel handles this correctly: threads assigned to tokens 5-31 load from valid
addresses (clamped by `lr1`) but the results are not written (the write loop at line
11501 only iterates `j < nr1`).

**Weight reads are shared**: the weight loading code at lines 11331-11352 loads weight
data regardless of how many tokens there are. Each weight element is read once into
threadgroup memory, then the simdgroup matmul uses it for all 5 tokens simultaneously.

**Conclusion: YES, the existing GEMM kernel shares weight reads at batch=5.**

At batch=5, weight bandwidth is reduced by ~4/5 = 80% compared to the GEMV path.

## Overhead Analysis

The GEMM path has overhead the GEMV path does not:

1. **Map kernel dispatch** (kernel_mul_mm_id_map0): one extra kernel launch + barrier.
   This scans all tokens for each expert. At batch=5, this is trivial (5 iterations per
   expert thread).

2. **Threadgroup memory setup**: 8KB shared memory (4KB sa + 4KB sb), loaded cooperatively
   by 128 threads. At batch=5, ~83% of sb loads are wasted (loading for 32 slots, only 5
   used).

3. **Simdgroup matmul**: Computes a full 64x32 tile even if only 5 columns are needed.
   ~84% of the compute is wasted.

4. **Wasted compute vs saved bandwidth**: For IQ3_S experts at [704, 2816]:
   - GEMV path: reads 1.75 MB per expert per token = 8.75 MB for 5 tokens
   - GEMM path: reads 1.75 MB per expert ONCE = 1.75 MB total (but with 6x compute overhead)
   - Memory bandwidth dominates on Apple Silicon (200 GB/s bandwidth, but expert reads
     go through mmap/SLC, effective bandwidth much lower for scattered reads)

## Optimal Threshold

| Threshold | batch=1 | batch=2-3 | batch=4-8 | batch=32+ |
|-----------|---------|-----------|-----------|-----------|
| 32 (current) | GEMV (fast) | GEMV (OK) | GEMV (slow: K reads) | GEMM (fast) |
| 4 (proposed) | GEMV (fast) | GEMV (OK) | GEMM (weight sharing) | GEMM (fast) |
| 2 (aggressive) | GEMV (fast) | GEMM (marginal) | GEMM (weight sharing) | GEMM (fast) |
| 1 (TQ default) | GEMM (overhead) | GEMM (marginal) | GEMM (weight sharing) | GEMM (fast) |

**Recommended: threshold = 4**

Rationale:
- batch=1 (autoregressive decode): GEMV is faster (no map kernel overhead, no wasted
  simdgroup compute). This is the hot path for normal generation.
- batch=2-3: Borderline. The 2-kernel overhead of GEMM (map0 + matmul) may not be worth
  the bandwidth saving for only 2-3 tokens. GEMV reads weights 2-3x but has zero setup.
- batch=4-8: Clear win for GEMM. Reading weights once instead of 4-8x saves 75-87.5%
  of expert weight bandwidth. The GEMM compute waste (computing 32-wide tiles for 4-8
  actual columns) is acceptable because MoE layers are bandwidth-bound, not compute-bound.
- batch=32+: Already uses GEMM.

## Expected Speedup for Speculative Decode (batch=5)

### MoE Layer Timing (batch=1, current GEMV)
- Each expert: ~1.75 MB read at ~100 GB/s effective = ~17.5 us
- 8 active experts per token = ~140 us per MoE layer
- batch=5 GEMV: 5 x 140 us = ~700 us per MoE layer

### MoE Layer Timing (batch=5, proposed GEMM)
- Weight read: 1.75 MB x 8 experts = 14 MB total (read once)
- At ~100 GB/s: ~140 us for weight reads
- Map kernel overhead: ~10-20 us
- Total: ~160-180 us per MoE layer

### Speedup: ~700 us / ~170 us = ~4.1x for MoE layers at batch=5

For the full model forward pass, MoE layers are ~60-70% of decode time, so overall
speculative decode verification would be ~2.5-3x faster.

## Implementation

Single line change in `ggml/src/ggml-metal/ggml-metal-ops.cpp` line 2420:

```c
// Before:
const int ne21_mm_id_min = is_tq_weight ? 1 : 32;

// After:
const int ne21_mm_id_min = is_tq_weight ? 1 : 4;
```

No kernel changes needed. The existing `kernel_mul_mm_id` handles small batch sizes
correctly -- it clamps tile dimensions and only writes valid output columns.

## Verification Checklist

Before deploying:
1. Correctness: Run model with threshold=4 and compare outputs to threshold=32 for
   batch sizes 1, 4, 5, 8.
2. Performance: Benchmark MUL_MAT_ID at batch=1 (must not regress) and batch=5 (must
   improve).
3. Edge cases: batch=4 exactly (boundary), batch=3 (should still use GEMV).

## Files

- Threshold: `ggml/src/ggml-metal/ggml-metal-ops.cpp:2420`
- GEMM kernel: `ggml/src/ggml-metal/ggml-metal.metal:11221` (kernel_mul_mm_id)
- Map kernel: `ggml/src/ggml-metal/ggml-metal.metal:11148` (kernel_mul_mm_id_map0)
- GEMV kernel: `ggml/src/ggml-metal/ggml-metal.metal:11792` (kernel_mul_mv_id)
