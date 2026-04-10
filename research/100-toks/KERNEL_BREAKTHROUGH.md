# BREAKTHROUGH: kernel_mul_mv_id Forces batch=1 Per Expert

## Discovery Date: 2026-04-10

## The Finding

The Metal MoE kernel `kernel_mul_mv_id` (line 11792 in ggml-metal.metal) 
HARDCODES `ne11=1` when calling the inner GEMV kernel, even when the batch
contains multiple tokens. This forces each (expert, token) pair to read
expert weights independently — NO weight sharing across batch elements.

Meanwhile, the inner kernel `kernel_mul_mv_ext_q4_f32_impl` ALREADY supports
batch reuse via the `r1ptg` template parameter. It dequantizes weights into
registers and computes K dot products. But kernel_mul_mv_id never uses this.

## Impact

For speculative decode with K=4 draft tokens:
- Current: 5 tokens × 8 experts = 40 threadgroups, each reading full expert weights
  → 40 × 2.3MB = 92MB weight reads per layer × 26 layers = 2.4GB per verification
- Fixed: 8 threadgroups, each reading weights ONCE for 5 tokens
  → 8 × 2.3MB = 18.4MB weight reads per layer × 26 layers = 478MB per verification
- Savings: 5x less data read = 5x faster MoE layers

## The Fix (Needs Careful Implementation)

1. In `kernel_mul_mv_id`: when `nei1 > 1`, change grid dispatch so z indexes
   only expert slots (8 values), not expert×token pairs (40 values)

2. Pass actual batch size in `ne11` field to inner kernel

3. Adjust src1/dst pointers to cover ALL tokens, not just one

4. The inner kernel's `r1ptg` loop at line 4570-4572 handles the rest:
   ```metal
   for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
       y4[ir1] = ...; // input vector for each token
   }
   ```

## Complexity

The challenge is coordinating:
- Grid dispatch dimensions (ggml-metal-ops.cpp)
- Memory access patterns in the kernel (ggml-metal.metal)
- The ids tensor layout (different tokens may activate different experts)
- The inner kernel's r1ptg template parameter selection

Estimated effort: 2-3 days of focused Metal kernel engineering.

## Expected Result

With this fix + speculative decode K=4:
- Verification batch reads expert weights ONCE for 5 tokens
- 5x reduction in MoE weight reads (60% of total model reads)
- Overall: ~3x speedup → 84 tok/s
- With K=8: ~6x reduction → potentially 100+ tok/s

This is the genuine path to 100 tok/s on 16GB Apple Silicon.
Nobody has done this because CUDA's per-kernel overhead is microseconds,
so the batch=1 path doesn't hurt on NVIDIA. On Metal with MoE models
on memory-constrained hardware, it's catastrophic.
