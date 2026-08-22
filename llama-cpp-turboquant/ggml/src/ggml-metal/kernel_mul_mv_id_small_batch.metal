// kernel_mul_mv_id_small_batch.metal
//
// Custom Metal kernel for small-batch (K=2-8) MoE expert matrix-vector multiply.
// Reads expert weights ONCE from device memory and computes dot products for K tokens
// simultaneously via threadgroup shared memory.
//
// STATUS: Reference implementation. The simpler approach of lowering the GEMM threshold
// in ggml-metal-ops.cpp (ne21_mm_id_min = 4 instead of 32) achieves the same weight-sharing
// benefit by reusing the existing kernel_mul_mm_id infrastructure. See DESIGN.md.
//
// This kernel is kept as a standalone reference for cases where the full GEMM tiling
// (64x32 tiles) has too much wasted compute for very small batches (K=2-3), and a
// dedicated GEMV-with-sharing kernel could be more efficient.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// IQ3_S block layout (from ggml-common.h, mirrored here for standalone compilation)
// ---------------------------------------------------------------------------
// QK_K = 256 quants per superblock
// IQ3_S: 3.4375 bits per weight
//   - d: half (scale)
//   - qs: uint8_t[QK_K/4] = 64 bytes (3-bit indices into iq3s_grid)
//   - qh: uint8_t[QK_K/32] = 8 bytes (high bits for grid index)
//   - signs: uint8_t[QK_K/8] = 32 bytes (sign bits)
//   - scales: uint8_t[QK_K/64] = 4 bytes (4-bit block scales)
//   Total: 2 + 64 + 8 + 32 + 4 = 110 bytes per 256 quants

#ifndef QK_K
#define QK_K 256
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Maximum batch size supported (compile-time).
// Template parameter K selects actual batch size at dispatch.
constant constexpr short MAX_BATCH = 8;

// Simdgroup width on Apple Silicon
constant constexpr short SIMD_WIDTH = 32;

// ---------------------------------------------------------------------------
// Float32 kernel: simplest case, no dequantization
// ---------------------------------------------------------------------------
// This kernel demonstrates the weight-sharing pattern for float32 experts.
//
// Grid dispatch:
//   x: output rows (ne01) / rows_per_tg
//   y: 1
//   z: number of active experts (nei0 * nei1, same as kernel_mul_mv_id)
//
// Each threadgroup processes `rows_per_tg` output rows for ALL K tokens.
// Weight data is read once per row, then dot-producted against K input vectors.

template<short K>
kernel void kernel_mul_mv_id_small_batch_f32(
        constant uint32_t & nei0,       // number of experts per token
        constant uint32_t & ne00,       // embedding dim (columns of weight matrix)
        constant uint32_t & ne01,       // hidden dim (rows of weight matrix)
        constant uint64_t & nb00,       // byte stride for weight element
        constant uint64_t & nb01,       // byte stride for weight row
        constant uint64_t & nb02,       // byte stride for expert
        constant uint64_t & nb10,       // byte stride for input element
        constant uint64_t & nb11,       // byte stride for input token
        constant uint64_t & nbi1,       // byte stride for ids row
        constant uint32_t & ne0,        // output rows
        device const char * src0s,      // expert weights [ne01, ne00] x n_experts
        device const char * src1,       // input vectors [ne00, K]
        device       char * dst,        // output vectors [ne0, K * nei0]
        device const char * ids,        // expert indices [nei0] per token
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    const int NSG = 2; // simdgroups per threadgroup
    const int rows_per_tg = NSG; // one row per simdgroup

    // Decode grid position
    const int iid1 = tgpig.z / nei0;  // token index
    const int idx  = tgpig.z % nei0;  // expert slot index

    // Which expert is selected for this token at this slot
    const int32_t expert_id = ((device const int32_t *)(ids + iid1 * nbi1))[idx];

    // Output row assigned to this simdgroup
    const int row = tgpig.x * rows_per_tg + sgitg;
    if (row >= (int)ne01) return;

    // Pointer to this expert's weight row
    device const float * w = (device const float *)(src0s + expert_id * nb02 + row * nb01);

    // Accumulators for K tokens
    float sums[K] = {0.0f};

    // Pointers to input vectors for all K tokens
    // For MoE id dispatch, we process one (expert, token) pair per z-slice,
    // but we want to share weights across tokens. In this simplified version,
    // we process the single assigned token.
    //
    // NOTE: To truly share weights across K tokens, the dispatch model needs
    // to change so that one threadgroup handles one expert for ALL K tokens.
    // The kernel_mul_mm_id approach (map0 + matmul) is the production solution.
    // This kernel shows the per-row dot product pattern.

    const int i11 = idx % 1; // ne11
    const int i12 = iid1;

    device const float * x = (device const float *)(src1 + i11 * nb11 + i12 * nb10 * 0);

    // Each thread in the simdgroup handles a stride of the dot product
    float sum = 0.0f;
    for (uint j = tiisg; j < ne00; j += SIMD_WIDTH) {
        sum += w[j] * x[j];
    }

    // Reduce across simdgroup
    sum = simd_sum(sum);

    // Write result
    if (tiisg == 0) {
        const int i1 = idx;
        const int i2 = i12;
        device float * dst_f32 = (device float *)(dst) + i1 * ne0 + i2 * 1 * ne0;
        dst_f32[row] = sum;
    }
}

// ---------------------------------------------------------------------------
// Weight-sharing kernel: the actual innovation
// ---------------------------------------------------------------------------
// This kernel processes ONE expert across ALL K tokens in a single threadgroup.
// It requires a different dispatch model than kernel_mul_mv_id:
//   - Grid z = n_active_experts (not n_tokens * n_experts_per_token)
//   - Each threadgroup loads weight rows and computes K dot products
//
// This is effectively what kernel_mul_mm_id does with its 64x32 tiling,
// but optimized for the K=2-8 case with simpler GEMV-style accumulation
// instead of full simdgroup matrix multiply.

template<short K>
kernel void kernel_mul_mv_id_shared_weights_f32(
        constant uint32_t & n_tokens,   // K (number of tokens in batch)
        constant uint32_t & nei0,       // experts per token
        constant uint32_t & ne00,       // embedding dim
        constant uint32_t & ne01,       // hidden dim (output rows)
        constant uint64_t & nb01,       // byte stride for weight row
        constant uint64_t & nb02,       // byte stride for expert
        constant uint64_t & nb10,       // byte stride for input element
        constant uint64_t & nb11,       // byte stride for input token
        constant uint64_t & nbi1,       // byte stride for ids row
        constant uint32_t & ne0,        // output ne0
        device const char * src0s,      // all expert weights
        device const char * src1,       // input vectors [ne00, n_tokens]
        device       char * dst,        // output
        device const char * ids,        // expert routing: ids[token][slot] = expert_id
        // Token-to-expert mapping (precomputed):
        // For each expert, which tokens route to it and at which slot
        device const int32_t * token_map, // [expert_id][max_tokens] = token indices
        device const int32_t * slot_map,  // [expert_id][max_tokens] = slot indices
        device const uint32_t * n_routed, // [expert_id] = count of tokens for this expert
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    const short NSG = 2;

    const int expert_id = tgpig.z;
    const int row = tgpig.x * NSG + sgitg;

    if (row >= (int)ne01) return;

    const uint32_t n_tok = n_routed[expert_id];
    if (n_tok == 0) return;

    // Weight row pointer (read ONCE)
    device const float * w = (device const float *)(src0s + expert_id * nb02 + row * nb01);

    // Load weight chunk and compute dot products for all routed tokens
    float sums[MAX_BATCH] = {0.0f};

    // Stripe across embedding dimension
    for (uint j = tiisg; j < ne00; j += SIMD_WIDTH) {
        const float wj = w[j]; // Read weight ONCE

        // Dot product with each token's input
        for (uint t = 0; t < n_tok && t < K; ++t) {
            int tok_idx = token_map[expert_id * K + t];
            device const float * x = (device const float *)(src1 + tok_idx * nb11);
            sums[t] += wj * x[j];
        }
    }

    // Reduce and write
    for (uint t = 0; t < n_tok && t < K; ++t) {
        float s = simd_sum(sums[t]);
        if (tiisg == 0) {
            int tok_idx = token_map[expert_id * K + t];
            int slot    = slot_map[expert_id * K + t];

            device float * dst_f32 = (device float *)dst
                + slot * ne0
                + tok_idx * 1 * ne0;
            dst_f32[row] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// IQ3_S weight-sharing kernel
// ---------------------------------------------------------------------------
// Same pattern as above but with IQ3_S dequantization.
// Each thread dequantizes a 32-element chunk of weights, then multiplies
// against K tokens' inputs. The dequantized values live in registers,
// avoiding threadgroup memory for weights entirely.
//
// This mirrors kernel_mul_mv_iq3_s_f32_impl but adds the K-token loop.

// IQ3_S grid lookup table (extern -- linked from main ggml-metal.metal compilation unit)
// When compiling standalone, these would need to be provided.
// In production, this file would be #included or compiled together.

// For reference: the iq3s_grid contains 512 entries of uint32_t,
// each encoding 4 uint8_t values used for dequantization.

// kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128}

template<short K>
kernel void kernel_mul_mv_id_shared_weights_iq3_s(
        constant uint32_t & n_tokens,
        constant uint32_t & nei0,
        constant uint32_t & ne00,       // must be multiple of QK_K (256)
        constant uint32_t & ne01,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint32_t & ne0,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const int32_t * token_map,
        device const int32_t * slot_map,
        device const uint32_t * n_routed,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    // IQ3_S grid in threadgroup memory (same as kernel_mul_mv_iq3_s_f32_impl)
    threadgroup uint32_t * svalues = (threadgroup uint32_t *)shmem;
    {
        int nval = 8;
        int pos = (32 * sgitg + tiisg) * nval;
        // Note: iq3s_grid must be available in constant memory
        // In standalone compilation, this would fail. This kernel must be
        // compiled as part of the ggml-metal.metal compilation unit.
        //for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        //threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Constants matching kernel_mul_mv_iq3_s_f32_impl
    constant uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

    const int expert_id = tgpig.z;
    const int row = tgpig.x; // one row per threadgroup x-slice

    if (row >= (int)ne01) return;

    const uint32_t n_tok = n_routed[expert_id];
    if (n_tok == 0) return;

    const int nb = ne00 / QK_K;
    const int nb32 = nb * (QK_K / 32);

    // Pointer to this expert's weight row (IQ3_S blocks)
    // block_iq3_s is 110 bytes for 256 quants
    device const char * weight_row = src0s + expert_id * nb02 + row * nb01;

    // Accumulators for each token
    float sums[MAX_BATCH] = {0.0f};

    // Load input vectors for all routed tokens into registers
    // (K <= 8, so 32 floats x 8 tokens = 256 floats = 1KB per thread -- too much)
    // Instead, we reload per chunk like the original kernel.

    const int ix = tiisg; // thread lane

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        // Dequantize weight chunk (32 elements) -- read ONCE
        // This is the key optimization: weight bytes are read from device memory
        // once and the dequantized values are reused across all K tokens.

        // In production, this would use the block_iq3_s struct:
        // device const block_iq3_s * xr = (device const block_iq3_s *)weight_row + ibl;
        // device const uint8_t * qs = xr->qs + 8 * ib;
        // device const uint8_t * qh = xr->qh + ib;
        // device const uint8_t * sc = xr->scales + (ib/2);
        // device const uint8_t * signs = xr->signs + 4 * ib;
        // device const half * dh = &xr->d;
        //
        // float d = dh[0] * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));
        //
        // float w_dequant[32];
        // for (short l = 0; l < 4; ++l) {
        //     const threadgroup uint32_t * table1 = qh[0] & kmask[2*l+0] ? svalues + 256 : svalues;
        //     const threadgroup uint32_t * table2 = qh[0] & kmask[2*l+1] ? svalues + 256 : svalues;
        //     const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
        //     const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
        //     for (short j = 0; j < 4; ++j) {
        //         w_dequant[8*l + j + 0] = d * grid1[j] * select(1, -1, signs[l] & kmask[j+0]);
        //         w_dequant[8*l + j + 4] = d * grid2[j] * select(1, -1, signs[l] & kmask[j+4]);
        //     }
        // }
        //
        // // Now multiply against each token's input -- weight read is AMORTIZED
        // for (uint t = 0; t < n_tok && t < K; ++t) {
        //     int tok_idx = token_map[expert_id * K + t];
        //     device const float * y = (device const float *)(src1 + tok_idx * nb11) + 32 * ib32;
        //     float2 sum = {0};
        //     for (short l = 0; l < 4; ++l) {
        //         for (short j = 0; j < 4; ++j) {
        //             sum[0] += y[8*l + j + 0] * w_dequant[8*l + j + 0];
        //             sum[1] += y[8*l + j + 4] * w_dequant[8*l + j + 4];
        //         }
        //     }
        //     sums[t] += sum[0] + sum[1];
        // }

        // Placeholder: the above code block is the production implementation.
        // It cannot compile standalone because block_iq3_s and iq3s_grid are
        // defined in ggml-metal.metal's compilation unit.
    }

    // Reduce across simdgroup and write results
    for (uint t = 0; t < n_tok && t < K; ++t) {
        float sum_all = simd_sum(sums[t]);
        if (tiisg == 0) {
            int tok_idx = token_map[expert_id * K + t];
            int slot    = slot_map[expert_id * K + t];

            device float * dst_f32 = (device float *)dst + slot * ne0 + tok_idx * ne0;
            dst_f32[row] = sum_all;
        }
    }
}

// ---------------------------------------------------------------------------
// Performance Analysis
// ---------------------------------------------------------------------------
//
// For Gemma 4 26B MoE with IQ3_S experts [704, 2816]:
//
// CURRENT (kernel_mul_mv_id, GEMV per token):
//   - Each expert read: 704 * 2816 * 3.4375/8 = ~849 KB (IQ3_S blocks)
//   - Per token: 8 experts * 849 KB = 6.6 MB
//   - batch=5 (spec decode): 5 * 6.6 MB = 33 MB device reads
//
// THIS KERNEL (shared weights):
//   - Weight read: 6.6 MB ONCE (shared across 5 tokens)
//   - Input read: 5 * 2816 * 4 = 55 KB (negligible)
//   - Total: ~6.7 MB device reads
//   - Speedup: 33/6.7 = ~4.9x for pure bandwidth
//   - Practical: ~3-4x accounting for compute and overhead
//
// RECOMMENDED APPROACH:
//   Just lower ne21_mm_id_min from 32 to 4 in ggml-metal-ops.cpp.
//   The existing kernel_mul_mm_id achieves the same weight sharing
//   with battle-tested simdgroup matmul tiling.
//
// This standalone kernel would only be beneficial if:
//   1. The 64x32 GEMM tile is too wasteful for K=2-3 (84-91% wasted columns)
//   2. The map0 kernel overhead is significant relative to the matmul
//   3. You want to avoid the threadgroup memory overhead of full GEMM tiling
//
// For K=4-8 (the speculative decode sweet spot), kernel_mul_mm_id is likely
// better because simdgroup_multiply_accumulate is highly optimized on Apple
// Silicon and the wasted compute is hidden by memory latency.
