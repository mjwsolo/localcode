# MOONSHOT: GPU-Autonomous Multi-Step Decode via Chained Argument Buffers

## The Core Insight

The GPU can do argmax, index into an embedding table, and feed the result into the next forward pass -- ALL without the CPU ever seeing the intermediate tokens. We encode K complete forward passes into a single Metal command buffer, connected by lightweight GPU kernels that bridge each step.

For greedy decoding, this produces EXACT results. No speculation, no verification, no draft model.

**Target: 4-8 tokens per Metal submission = 100-200 tok/s on M4.**

## Why This Has Never Been Done

Every LLM inference engine (llama.cpp, vLLM, TensorRT-LLM, MLX) follows the same pattern inherited from training frameworks:

```
loop:
    logits = forward(token)      # GPU
    next_token = sample(logits)  # CPU
    token = next_token           # CPU
```

The CPU-side sampling exists because:
1. Training frameworks did it this way (historical inertia)
2. CUDA doesn't have the same per-submission overhead problem (CUDA streams are ~5us overhead)
3. Sampling strategies (temperature, top-p, nucleus) are traditionally CPU code
4. Nobody working on Apple Silicon inference has been desperate enough to break this pattern

But on Metal, where the submission overhead is 35ms, this CPU round-trip is catastrophic. The GPU finishes in ~3ms and then waits 32ms for the CPU to give it the next token.

## Architecture

### Memory Layout

```
GPU Buffers (all MTLResourceStorageModeShared):
  [embedding_table]  256K tokens * 3584 dims * fp16 = 1.75 GB (dequantized from GGUF at init)
                     OR: keep IQ3_S and dequantize on GPU per step = 0 extra memory
  [kv_cache]         Existing turbo4 KV cache, 355 MiB
  [step_tokens]      K * sizeof(int32)  -- token IDs for each sub-step
  [step_logits]      K * vocab_size * sizeof(fp16) -- optional, for CPU to read final distribution
  [control_block]    Small buffer with step count, sampling params, stop tokens
```

### Command Buffer Structure

For K=4 (four tokens per submission):

```
COMMAND BUFFER:
  |-- Encoder 0: Forward Pass Step 0
  |   (reads token embedding from step_tokens[0], writes logits_0)
  |   (writes KV cache at position P)
  |
  |-- Memory Barrier
  |
  |-- Encoder 1: Bridge Kernel 0->1
  |   kernel bridge_step(logits_0, embedding_table, step_tokens, control_block):
  |     token_id = argmax(logits_0)          // greedy
  |     step_tokens[1] = token_id
  |     if token_id == EOS: control_block.stop = 1
  |
  |-- Memory Barrier
  |
  |-- Encoder 2: Forward Pass Step 1
  |   (reads token embedding from step_tokens[1], writes logits_1)
  |   (writes KV cache at position P+1)
  |   (BUT: if control_block.stop == 1, this is wasted compute)
  |
  |-- Memory Barrier
  |
  |-- Encoder 3: Bridge Kernel 1->2
  |   (same as Bridge Kernel 0->1, reading logits_1, writing step_tokens[2])
  |
  |-- Encoder 4: Forward Pass Step 2
  |   ...
  |
  |-- Encoder 5: Bridge Kernel 2->3
  |   ...
  |
  |-- Encoder 6: Forward Pass Step 3
  |   ...
  |
  |-- Encoder 7: Final Bridge
  |   kernel final_bridge(logits_3, step_tokens, control_block):
  |     step_tokens[4] = argmax(logits_3)    // for CPU to know the next token
  |     control_block.steps_completed = actual_steps_before_EOS
```

### The Bridge Kernel (Metal Shader)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void bridge_argmax_embed(
    device const half    * logits         [[buffer(0)]],  // [vocab_size]
    device const half    * embedding_table [[buffer(1)]],  // [vocab_size, hidden_dim]
    device       int     * step_tokens    [[buffer(2)]],  // [K+1]
    device       half    * next_input     [[buffer(3)]],  // [hidden_dim] -- pre-RMSNorm embedding
    constant     uint    & vocab_size     [[buffer(4)]],
    constant     uint    & hidden_dim     [[buffer(5)]],
    constant     uint    & step_idx       [[buffer(6)]],
    device       int     * control        [[buffer(7)]],  // [stop_flag, eos_token, steps_done]
    uint tid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    // Phase 1: Parallel argmax reduction (first threadgroup only)
    if (tg_id == 0) {
        // Each thread finds local max in its chunk
        threadgroup int local_max_idx[256];
        threadgroup half local_max_val[256];

        int chunk = (vocab_size + tg_size - 1) / tg_size;
        int start = tid * chunk;
        int end = min(start + chunk, vocab_size);

        int best_idx = start;
        half best_val = logits[start];
        for (int i = start + 1; i < end; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        local_max_idx[tid] = best_idx;
        local_max_val[tid] = best_val;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 does final reduction
        if (tid == 0) {
            int global_best = local_max_idx[0];
            half global_val = local_max_val[0];
            for (uint i = 1; i < tg_size; i++) {
                if (local_max_val[i] > global_val) {
                    global_val = local_max_val[i];
                    global_best = local_max_idx[i];
                }
            }

            step_tokens[step_idx + 1] = global_best;

            // Check for EOS
            if (global_best == control[1]) {
                control[0] = 1;  // stop flag
                control[2] = step_idx + 1;  // steps completed
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: Copy embedding for next step (all threadgroups)
    // Each thread copies one element of the embedding
    int token_id = step_tokens[step_idx + 1];
    uint elem_idx = tg_id * tg_size + tid;
    if (elem_idx < hidden_dim) {
        next_input[elem_idx] = embedding_table[token_id * hidden_dim + elem_idx];
    }
}
```

### The EOS Problem

If the model generates EOS at step 1 of 4, steps 2-3 are wasted compute. Solutions:

1. **Accept the waste**: At K=4, worst case wastes 3 forward passes. But the overhead savings (3 * 35ms = 105ms saved) far exceed the wasted compute (3 * 3ms = 9ms). Net gain: 96ms.

2. **Short-circuit via control block**: Each forward pass checks `control_block.stop` at the start and early-exits. Metal doesn't support true early exit from a command buffer, but each kernel can check the flag and do nothing (skip all writes). This wastes GPU cycles but not memory bandwidth.

3. **Statistical K selection**: Measure average generation length and set K = min(4, expected_remaining). For code generation, sequences are typically long, so K=4-8 wastes very little.

### The KV Cache Challenge

Each sub-step within the command buffer must write to the correct KV cache position. Since we know the positions in advance (P, P+1, P+2, P+3), we can pre-compute the KV cache offsets and pass them as constants to each forward pass.

**Critical detail**: The KV cache position for step N+1 must account for step N's write. With turbo4 quantization, the write positions are deterministic -- we just increment the position counter. The graph builder already handles this via `inp_pos` tensor. We pre-fill it with [P, P+1, P+2, P+3].

### Modifying llama.cpp's Graph Builder

The key change is in `llama_context::process()` (the decode path):

```cpp
// CURRENT: builds one forward pass graph
gf = model.build_graph(gparams);  // one token

// NEW: builds K chained forward pass graphs
for (int step = 0; step < K; step++) {
    auto gf_step = model.build_graph(gparams_step[step]);

    if (step > 0) {
        // Insert bridge kernel between step-1 and step
        auto bridge = ggml_custom_op(ctx, "bridge_argmax_embed",
            logits_prev, embedding_table, step_tokens, ...);
        ggml_build_forward_expand(gf, bridge);
    }

    // Append this step's nodes to the master graph
    for (int i = 0; i < gf_step->n_nodes; i++) {
        ggml_build_forward_expand(gf, gf_step->nodes[i]);
    }

    logits_prev = gf_step->output_logits;
}
```

This creates ONE large graph with K forward passes + (K-1) bridge kernels. The Metal backend encodes all of it into command buffers and submits.

### The Embedding Table

Two options:

**Option A: Pre-dequantize at init (costs ~1.75GB)**
At startup, dequantize the entire `token_embd` tensor from IQ3_S to fp16 and store in a separate MTLBuffer. Fast lookup but costs memory.

**Option B: On-the-fly dequantize (costs 0 extra memory)**
The bridge kernel reads the quantized embedding and dequantizes inline. IQ3_S dequantization is not trivial but is well-defined. Since we only dequantize ONE embedding (3584 values) per bridge, the cost is negligible (~1us).

**Recommendation**: Option B. Memory is precious on 16GB.

### Non-Greedy Sampling on GPU

For temperature + top-p sampling:

```metal
kernel void bridge_sample(
    device const half * logits,
    device half * probs,          // scratch buffer
    constant float & temperature,
    constant float & top_p,
    device uint * rng_state,      // xoshiro256** state
    ...
) {
    // 1. Apply temperature: probs[i] = logits[i] / temperature
    // 2. Softmax (parallel reduction)
    // 3. Sort descending (bitonic sort on GPU, or just find top-K with selection)
    // 4. Cumulative sum until > top_p
    // 5. Renormalize
    // 6. Draw from uniform using xoshiro256**
    // 7. Binary search CDF for token_id
}
```

GPU-side sorting of 256K elements is well-studied (bitonic sort: O(n log^2 n) parallel). For top-p, we often only need top-50 or so, so a partial sort / selection algorithm suffices.

The RNG state lives in GPU memory and is seeded once from CPU. Each step advances the state.

## Implementation Plan

### Phase 1: Proof of Concept (Greedy Only, K=2)
1. Modify `llama_model::build_graph` to accept a `n_decode_steps` parameter
2. Build the bridge kernel (argmax only, skip sampling)
3. Build two forward passes connected by the bridge into one graph
4. Test correctness: output must match single-step decode
5. Benchmark: expect ~50-56 tok/s (35ms / 2 + small GPU compute overhead)

### Phase 2: Scale to K=4-8
1. Generalize to K steps
2. Add EOS detection and waste minimization
3. Handle KV cache position management for K steps
4. Benchmark: expect 80-150 tok/s

### Phase 3: GPU-Side Sampling
1. Implement temperature + top-p on GPU
2. Add xoshiro256** RNG to bridge kernel
3. Validate sampling distribution matches CPU sampling
4. Full production deployment

### Phase 4: Adaptive K
1. Monitor actual generation length and EOS frequency
2. Dynamically adjust K to minimize waste
3. Use K=1 for the last few tokens of a generation (when EOS is likely)

## Risk Analysis

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| KV cache position errors | Medium | Extensive testing with known outputs |
| Memory pressure from larger graphs | Low | K=4 only 4x the nodes, graph memory is small vs model |
| GPU compute becoming bottleneck at K=8 | Medium | Profile and cap K where throughput plateaus |
| EOS waste for short outputs | Low | Adaptive K handles this |
| Metal command buffer size limit | Low | Even at K=8, we have ~3200 dispatches, well under Metal's limits |
| Correctness of GPU-side sampling | Medium | Statistical tests comparing GPU vs CPU sampling distributions |

## Why This Is The Moonshot

Nobody in the LLM inference community has built GPU-autonomous multi-step autoregressive decoding. The reason is simple: on CUDA, the per-submission overhead is ~5us, so there's no motivation. On Metal, the 35ms overhead makes it essential.

This approach turns Metal's biggest weakness (high submission latency) into an irrelevance by simply not submitting between tokens. The GPU becomes a self-contained token generation machine that only returns to the CPU every K tokens.

If K=8 works reliably: **35ms / 8 = 4.4ms per token = 227 tok/s.**

That's not just hitting 100 tok/s. That's blowing past it. On a laptop. With no cloud. With a 26B parameter model.

That would be a genuine breakthrough in local inference.
