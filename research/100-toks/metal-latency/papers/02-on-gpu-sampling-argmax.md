# On-GPU Sampling / Argmax: Eliminating the CPU Roundtrip

## Sources
- [llama.cpp PR #17004: Backend Sampling Support](https://github.com/ggml-org/llama.cpp/pull/17004)
- [llama.cpp Discussion #17621: Optimizing Token Generation](https://github.com/ggml-org/llama.cpp/discussions/17621)
- [MLX Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)

## The CPU Roundtrip Problem

Current llama.cpp decode loop per token:
```
1. GPU: compute forward pass -> produce logits tensor on GPU
2. CPU: llama_get_logits_ith() calls synchronize() [BLOCKS - 35ms]
3. CPU: ggml_backend_tensor_get_async() copies logits to CPU
4. CPU: sampling (argmax/top-k/top-p) selects next token
5. CPU: set next token as input for next decode
6. GOTO 1
```

Step 2 is the killer. `llama_get_logits()` and `llama_get_logits_ith()` both call
`ctx->synchronize()` which calls `ggml_backend_sched_synchronize()` which calls
`ggml_metal_synchronize()` which calls `[cmd_buf_last waitUntilCompleted]`.

## llama.cpp Backend Sampling (PR #17004, Merged Jan 2026)

This PR adds GPU-accelerated sampling as part of the computation graph:

### What it does:
- Implements GPU kernels for top-k, top-p, min-p, temperature, greedy selection
- Sampling runs ON the GPU as ggml operations in the computation graph
- Only the final sampled token ID needs to transfer to CPU (4 bytes vs N_VOCAB*4 bytes)
- Supports partial GPU sampling (compatible ops on GPU, incompatible on CPU)

### Metal support status:
- `GGML_OP_ARGMAX` is ALREADY implemented in Metal backend (confirmed in source)
- `GGML_OP_CUMSUM` is implemented in Metal (needed for top-p)
- Top-k Metal optimization is listed as "future work"
- The infrastructure is there but Metal-specific optimizations are incomplete

### Key limitation for our problem:
Backend sampling reduces DATA TRANSFER (logits -> CPU) but does NOT eliminate
the synchronize() call. The token ID still needs to come back to CPU to:
1. Update the KV cache position
2. Feed back as input to the next decode
3. Stream to the user

So backend sampling alone does NOT solve the 35ms problem.

## The Real Solution: On-GPU Autoregressive Loop

What we actually need is to run MULTIPLE decode steps in a single command buffer
submission, performing argmax on-GPU between steps:

```
Single command buffer:
  1. Forward pass (token N) -> logits
  2. Argmax on GPU -> token N+1 (stays on GPU)
  3. Forward pass (token N+1) -> logits
  4. Argmax on GPU -> token N+2 (stays on GPU)
  ...
  K. Forward pass (token N+K-1) -> logits
  K+1. Copy all K token IDs to CPU
```

This amortizes the 35ms commit overhead across K tokens.

### Implementation challenges:
1. **KV cache management**: Each step modifies the KV cache. The cache update
   must happen on-GPU between steps.
2. **Token embedding lookup**: Need the embedding for token N+1 to run step 2.
   This requires an embedding lookup kernel on GPU.
3. **Stop conditions**: EOS token must be detected on-GPU to break the loop.
4. **Graph construction**: The ggml graph must be built for K steps, not 1.
   This means K copies of the model graph chained together.

### What Metal supports:
- ARGMAX: Already in Metal backend
- Embedding lookup: Can be done with a gather/index operation
- KV cache update: Already happens on GPU during forward pass
- The main missing piece: building a K-step chained graph

## MLX's Approach

MLX uses lazy evaluation + async_eval to pipeline token generation:

```python
def generator():
    with mx.stream(mx.new_stream(mx.gpu)):
        out = mx.async_eval(my_function())
        while True:
            out_next = mx.async_eval(my_function())
            mx.eval(out)  # wait for previous
            yield out
            out = out_next
```

This overlaps GPU compute for token N+1 with CPU processing of token N.
MLX achieves 230 tok/s on M2 Ultra with this approach for some models.

But this is PIPELINING, not BATCHING. It still has one commit per token.
The reason it works for MLX is that MLX's Metal integration has lower
per-commit overhead than llama.cpp's (likely due to simpler resource management
and lazy evaluation reducing command buffer count).
