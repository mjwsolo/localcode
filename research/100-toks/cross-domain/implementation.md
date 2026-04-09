# Implementation: GPU-Driven Autoregressive Loop

The #1 ranked idea: move the entire autoregressive sampling loop to the GPU so multiple tokens are generated per Metal command buffer commit, amortizing the 35ms scheduling overhead.

## Architecture Overview

```
BEFORE (current):
  CPU: encode_graph -> commit -> [35ms] -> GPU: decode(T) -> [0ms] -> CPU: sync -> sample -> repeat
  Total: 35ms per token = 28 tok/s

AFTER (GPU-driven):
  CPU: encode_N_iterations -> commit -> [35ms] -> GPU: decode(T) -> argmax -> embed -> decode(T+1) -> argmax -> ... -> [N*0.5ms]
  CPU: read_back_N_tokens
  Total: 35ms + N*0.5ms per N tokens. At N=16: 43ms/16 = 2.7ms/token = ~370 tok/s (memory-BW limited to ~100)
```

## Phase A: GPU-Side Argmax Kernel

### Metal Compute Shader

```metal
// gpu_argmax.metal
// Performs argmax over logits buffer, writes token ID to output buffer.
// Uses parallel reduction: each threadgroup finds local max, then atomic global max.

#include <metal_stdlib>
using namespace metal;

// Stage 1: Per-threadgroup reduction
kernel void argmax_reduce(
    device const float*  logits      [[buffer(0)]],   // [vocab_size] logits
    device       uint*   partial_idx [[buffer(1)]],   // [n_threadgroups] partial results (index)
    device       float*  partial_val [[buffer(2)]],   // [n_threadgroups] partial results (value)
    constant     uint&   vocab_size  [[buffer(3)]],
    uint                 tid         [[thread_index_in_threadgroup]],
    uint                 tgid        [[threadgroup_position_in_grid]],
    uint                 tg_size     [[threads_per_threadgroup]]
) {
    // Shared memory for reduction within threadgroup
    threadgroup float shared_val[1024];
    threadgroup uint  shared_idx[1024];

    uint global_idx = tgid * tg_size + tid;

    // Initialize with -infinity
    float max_val = -INFINITY;
    uint  max_idx = 0;

    // Each thread handles multiple elements (grid-stride loop)
    for (uint i = global_idx; i < vocab_size; i += tg_size * gridDim.x) {
        float v = logits[i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    shared_val[tid] = max_val;
    shared_idx[tid] = max_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes threadgroup result
    if (tid == 0) {
        partial_val[tgid] = shared_val[0];
        partial_idx[tgid] = shared_idx[0];
    }
}

// Stage 2: Final reduction (runs with 1 threadgroup)
kernel void argmax_final(
    device const uint*   partial_idx   [[buffer(0)]],
    device const float*  partial_val   [[buffer(1)]],
    device       uint*   output_token  [[buffer(2)]],   // single uint: the winning token ID
    constant     uint&   n_partials    [[buffer(3)]],
    uint                 tid           [[thread_index_in_threadgroup]]
) {
    threadgroup float shared_val[256];
    threadgroup uint  shared_idx[256];

    float max_val = -INFINITY;
    uint  max_idx = 0;

    for (uint i = tid; i < n_partials; i += 256) {
        if (partial_val[i] > max_val) {
            max_val = partial_val[i];
            max_idx = partial_idx[i];
        }
    }

    shared_val[tid] = max_val;
    shared_idx[tid] = max_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && shared_val[tid + stride] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + stride];
            shared_idx[tid] = shared_idx[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output_token[0] = shared_idx[0];
    }
}
```

### Embedding Lookup Kernel

```metal
// gpu_embed_lookup.metal
// Reads token ID from argmax output, copies embedding row to input buffer.

kernel void embed_lookup(
    device const uint*   token_id   [[buffer(0)]],   // [1] from argmax output
    device const half*   embed_wt   [[buffer(1)]],   // [vocab_size, embed_dim] embedding matrix
    device       half*   output     [[buffer(2)]],   // [embed_dim] input for next decode
    constant     uint&   embed_dim  [[buffer(3)]],
    uint                 tid        [[thread_index_in_threadgroup]],
    uint                 tgid       [[threadgroup_position_in_grid]],
    uint                 tg_size    [[threads_per_threadgroup]]
) {
    uint tok = token_id[0];
    uint offset = tok * embed_dim;

    // Each thread copies a slice of the embedding
    uint idx = tgid * tg_size + tid;
    if (idx < embed_dim) {
        output[idx] = embed_wt[offset + idx];
    }
}
```

### EOS Detection Kernel

```metal
// gpu_eos_check.metal
// Checks if the generated token is EOS. Writes to shared memory for CPU polling.

kernel void eos_check(
    device const uint*   token_id       [[buffer(0)]],
    device       uint*   eos_flag       [[buffer(1)]],   // MTLStorageModeShared — CPU can read
    device       uint*   token_log      [[buffer(2)]],   // ring buffer of generated tokens
    device       uint*   token_count    [[buffer(3)]],   // atomic counter
    constant     uint&   eos_token      [[buffer(4)]],   // EOS token ID (e.g., 1)
    constant     uint&   max_tokens     [[buffer(5)]],   // safety limit
    uint                 tid            [[thread_index_in_threadgroup]]
) {
    if (tid == 0) {
        uint tok = token_id[0];
        uint count = token_count[0];

        // Write token to log
        token_log[count] = tok;
        token_count[0] = count + 1;

        // Check termination
        if (tok == eos_token || count + 1 >= max_tokens) {
            eos_flag[0] = 1;
        }
    }
}
```

## Phase B: Multi-Token Command Buffer Encoding

### Pseudocode for Modified `ggml_metal_graph_compute`

```objc
// Modified ggml_metal_graph_compute in ggml-metal-context.m
// Encodes N autoregressive iterations into ONE command buffer

enum ggml_status ggml_metal_graph_compute_multi(
    ggml_metal_t ctx,
    struct ggml_cgraph *gf,
    int n_iterations,                    // how many tokens to generate in one commit
    id<MTLBuffer> logits_buf,            // output logits buffer
    id<MTLBuffer> embed_weights_buf,     // embedding weight matrix
    id<MTLBuffer> input_embed_buf,       // input embedding buffer (mutable)
    id<MTLBuffer> token_output_buf,      // shared buffer: generated token IDs
    id<MTLBuffer> eos_flag_buf,          // shared buffer: EOS detection flag
    id<MTLBuffer> token_count_buf,       // shared buffer: token counter
    uint32_t embed_dim,
    uint32_t vocab_size,
    uint32_t eos_token_id,
    uint32_t max_tokens
) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];

        // Intermediate buffers for argmax reduction
        id<MTLBuffer> partial_idx = /* pre-allocated */;
        id<MTLBuffer> partial_val = /* pre-allocated */;
        id<MTLBuffer> argmax_out  = /* pre-allocated, single uint */;

        uint32_t n_threadgroups_argmax = (vocab_size + 1023) / 1024;

        for (int iter = 0; iter < n_iterations; iter++) {

            // === Step 1: Encode the full decode graph ===
            // This is the existing encode_async logic, but we modify the
            // KV cache offset to be (base_pos + iter) instead of (base_pos)
            {
                id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

                // Update KV cache position for this iteration
                uint32_t kv_pos = ctx->base_kv_pos + iter;
                // ... encode all graph nodes with updated kv_pos ...
                // (reuse existing ggml_metal_encode_node logic)

                for (int node_idx = 0; node_idx < gf->n_nodes; node_idx++) {
                    ggml_metal_encode_node(ctx, enc, gf->nodes[node_idx], kv_pos);
                }

                [enc endEncoding];
            }

            // === Step 2: GPU-side argmax on logits ===
            {
                id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

                [enc setComputePipelineState:ctx->pipeline_argmax_reduce];
                [enc setBuffer:logits_buf      offset:0 atIndex:0];
                [enc setBuffer:partial_idx     offset:0 atIndex:1];
                [enc setBuffer:partial_val     offset:0 atIndex:2];
                [enc setBytes:&vocab_size      length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(n_threadgroups_argmax, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

                [enc setComputePipelineState:ctx->pipeline_argmax_final];
                [enc setBuffer:partial_idx     offset:0 atIndex:0];
                [enc setBuffer:partial_val     offset:0 atIndex:1];
                [enc setBuffer:argmax_out      offset:0 atIndex:2];
                uint32_t n_tg = n_threadgroups_argmax;
                [enc setBytes:&n_tg            length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc endEncoding];
            }

            // === Step 3: EOS check + token logging ===
            {
                id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

                [enc setComputePipelineState:ctx->pipeline_eos_check];
                [enc setBuffer:argmax_out      offset:0 atIndex:0];
                [enc setBuffer:eos_flag_buf    offset:0 atIndex:1];
                [enc setBuffer:token_output_buf offset:0 atIndex:2];
                [enc setBuffer:token_count_buf offset:0 atIndex:3];
                [enc setBytes:&eos_token_id    length:4 atIndex:4];
                [enc setBytes:&max_tokens      length:4 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                [enc endEncoding];
            }

            // === Step 4: Embedding lookup for next iteration ===
            if (iter < n_iterations - 1) {
                id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

                [enc setComputePipelineState:ctx->pipeline_embed_lookup];
                [enc setBuffer:argmax_out       offset:0 atIndex:0];
                [enc setBuffer:embed_weights_buf offset:0 atIndex:1];
                [enc setBuffer:input_embed_buf  offset:0 atIndex:2];
                [enc setBytes:&embed_dim        length:4 atIndex:3];

                uint32_t n_tg_embed = (embed_dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(n_tg_embed, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc endEncoding];
            }

            // NOTE: Metal automatically sequences compute encoders within
            // the same command buffer. No explicit fence needed — each
            // encoder waits for the previous one to complete.
        }

        // === Commit once for all N iterations ===
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        // === CPU reads results from shared buffers ===
        uint32_t *tokens = (uint32_t *)[token_output_buf contents];
        uint32_t *count  = (uint32_t *)[token_count_buf contents];
        uint32_t *eos    = (uint32_t *)[eos_flag_buf contents];

        // Return generated tokens to caller
        // tokens[0..count[0]-1] are the generated token IDs
        // eos[0] indicates if EOS was hit

        return GGML_STATUS_SUCCESS;
    }
}
```

## Phase C: Integration with llama.cpp Decode Loop

### Modified decode path in llama-context.cpp

```cpp
// Pseudocode for multi-token decode in llama-context.cpp

// New function: decode N tokens autoregressively on GPU
int llama_decode_multi(
    llama_context * ctx,
    llama_token     first_token,
    int             n_tokens_max,    // max tokens to generate per commit
    llama_token *   output_tokens,   // output: generated tokens
    int *           n_generated,     // output: how many tokens generated
    bool *          hit_eos          // output: did we hit EOS?
) {
    // 1. Set up the first token's input
    llama_batch batch = llama_batch_get_one(&first_token, 1);
    
    // 2. Build the compute graph (once — it's the same for all iterations)
    auto * gf = llama_build_graph(ctx, batch, /* n_tokens */ 1);
    
    // 3. Allocate shared buffers for GPU-CPU communication
    // These are MTLStorageModeShared — both CPU and GPU can access
    auto * token_buf = metal_alloc_shared(n_tokens_max * sizeof(uint32_t));
    auto * count_buf = metal_alloc_shared(sizeof(uint32_t));
    auto * eos_buf   = metal_alloc_shared(sizeof(uint32_t));
    
    // Zero out
    memset(token_buf, 0, n_tokens_max * sizeof(uint32_t));
    memset(count_buf, 0, sizeof(uint32_t));
    memset(eos_buf, 0, sizeof(uint32_t));
    
    // 4. Submit multi-iteration compute to GPU
    ggml_metal_graph_compute_multi(
        ctx->metal_ctx, gf,
        n_tokens_max,
        ctx->logits_buffer,
        ctx->model->tok_embd,        // embedding weight matrix
        ctx->input_embed_buffer,     // mutable input
        token_buf, eos_buf, count_buf,
        ctx->model->hparams.n_embd,  // embed_dim = 4608 for Gemma 4
        ctx->model->hparams.n_vocab, // vocab_size = 262144
        ctx->model->vocab.token_eos, // EOS token ID
        n_tokens_max
    );
    
    // 5. Read results
    *n_generated = ((uint32_t *)count_buf)[0];
    *hit_eos = ((uint32_t *)eos_buf)[0] != 0;
    memcpy(output_tokens, token_buf, *n_generated * sizeof(llama_token));
    
    // 6. Update KV cache position
    ctx->kv_self.head += *n_generated;
    
    return 0;
}
```

## Phase D: Streaming Output to User

The user needs to see tokens as they're generated, not wait for the entire batch. Since the GPU generates tokens into a shared memory buffer, the CPU can poll it:

```cpp
// In the server/CLI response streaming loop:

void stream_multi_token_decode(llama_context * ctx, llama_token first_token) {
    const int BATCH_SIZE = 16;  // tokens per GPU commit
    llama_token output[BATCH_SIZE];
    int n_generated;
    bool hit_eos;
    uint32_t last_read = 0;

    while (!hit_eos) {
        // Submit batch to GPU (non-blocking would be even better,
        // but waitUntilCompleted is fine since GPU does all the work)
        llama_decode_multi(ctx, first_token, BATCH_SIZE,
                          output, &n_generated, &hit_eos);

        // Stream all generated tokens to output
        for (int i = 0; i < n_generated; i++) {
            send_token_to_user(output[i]);
        }

        // The LAST generated token becomes the first token of next batch
        // (only needed if we didn't hit EOS)
        if (!hit_eos && n_generated > 0) {
            first_token = output[n_generated - 1];
        }
    }
}
```

## Latency Analysis

### Per-token overhead breakdown with N=16

| Component | Time | Per token (N=16) |
|-----------|------|-------------------|
| Command buffer commit + Metal scheduling | 35ms | 2.2ms |
| GPU decode (26 MoE layers, 1 token) | ~0.5ms | 0.5ms |
| GPU argmax (262K vocab parallel reduce) | ~0.05ms | 0.05ms |
| GPU embed lookup (4608 half-precision) | ~0.01ms | 0.01ms |
| GPU EOS check | ~0.001ms | 0.001ms |
| **Total per token** | | **~2.8ms** |
| **Throughput** | | **~360 tok/s** |

Memory bandwidth limit check:
- Per token: read ~1.2GB of active expert weights (top-8 of 128, each ~150MB)
- M4 bandwidth: ~100 GB/s
- Time to read: 1.2GB / 100 GB/s = 12ms
- With N=16: 16 * 12ms = 192ms GPU time + 35ms scheduling = 227ms / 16 = 14.2ms/tok
- **Memory-BW limited throughput: ~70 tok/s**

Wait — the 0.5ms GPU time above was from ROOT_CAUSE.md (measured with actual workload). Let me reconcile:
- ROOT_CAUSE says GPU compute is ~0ms. But prompt eval at 318 tok/s suggests ~3ms per token in batch.
- The "0ms" is likely because Metal timestamps only capture kernel time, not memory reads.
- True per-token GPU time (including memory) is ~3ms (from 318 tok/s prompt eval).

Revised with 3ms/token GPU time:
| Component | Time | Per token (N=16) |
|-----------|------|-------------------|
| Metal scheduling | 35ms | 2.2ms |
| GPU decode + memory reads | 3ms | 3.0ms |
| GPU argmax + embed + EOS | 0.1ms | 0.1ms |
| **Total per token** | | **~5.3ms** |
| **Throughput** | | **~190 tok/s** |

Even conservative: **~100 tok/s with N=8** (35ms/8 + 3ms = 7.4ms/tok = 135 tok/s).

## Known Limitations and TODO

1. **Greedy-only sampling**: This implementation uses argmax (greedy). For temperature/top-k/top-p, need a GPU-side sampling kernel with random number generation (Metal has `metal::random`). This is a straightforward extension but not trivial.

2. **EOS in middle of batch**: If EOS occurs at token 5 of a 16-token batch, tokens 6-15 are wasted computation. Mitigation: use smaller batches (N=8) or implement early termination via indirect command buffer conditional execution.

3. **Token streaming latency**: User sees tokens in bursts of N instead of one at a time. With N=8 at 100 tok/s, burst arrives every 60ms — acceptable (below 100ms perception threshold). Could reduce N=4 for more responsive streaming.

4. **KV cache offset**: Each iteration within the batch needs a different KV cache position. These are sequential and known in advance. Encode them as constants in the command buffer or as an array in a parameter buffer.

5. **Graph re-encoding**: Currently the graph is encoded fresh each time. With multi-token batching, we encode N copies of the same graph. This CPU encoding time (0.3ms * N) adds up. Future optimization: use ICBs (#3 solution) to encode once and replay.

## Files to Modify

1. **New file**: `ggml/src/ggml-metal/gpu-sampling.metal` — argmax, embed lookup, EOS check kernels
2. **Modify**: `ggml/src/ggml-metal/ggml-metal-context.m` — add `ggml_metal_graph_compute_multi()`
3. **Modify**: `ggml/src/ggml-metal/ggml-metal-context.h` — declare new function, add pipeline state objects
4. **Modify**: `src/llama-context.cpp` — add `llama_decode_multi()` path
5. **Modify**: `include/llama.h` — expose new API
6. **Modify**: `examples/server/server.cpp` — use multi-token decode in completion endpoint
