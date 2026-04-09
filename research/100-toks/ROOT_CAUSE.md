# Root Cause: 35ms Metal Command Buffer Scheduling Latency

## Confirmed Finding (2025-04-09)

The decode speed bottleneck is a **35ms per-commit Metal scheduling overhead** that cannot be reduced by changing the wait mechanism.

### Evidence
- `waitUntilCompleted`: 35ms per token → 28 tok/s
- Spin-wait polling: 35ms per token → 26 tok/s (same latency, worse due to CPU burning)
- `GPUStartTime` to `GPUEndTime`: 0.0ms (GPU compute is instantaneous)
- Prompt eval (batched): 3.1ms per token → 318 tok/s (proves batching amortizes the 35ms)

### What the 35ms IS
Metal command buffer lifecycle:
1. Commit → enters Metal's scheduling queue
2. Metal sets up GPU memory mappings for mmap'd tensors (~10.4GB model)
3. Metal dispatches kernels to GPU
4. GPU executes kernels (~0ms for single-token decode)
5. Metal signals completion
6. Total: ~35ms

### What the 35ms is NOT
- NOT `waitUntilCompleted` overhead (spin-wait gives same result)
- NOT GPU compute time (0.0ms measured)
- NOT CPU graph build time (0.3ms measured)
- NOT memory bandwidth (prompt eval achieves 318 tok/s)

### Why Prompt Eval is Fast
Prompt eval submits hundreds of tokens in one batch → one commit → one 35ms overhead amortized across all tokens. Per-token cost: 35ms/500 = 0.07ms overhead.

### Only Path to 100 tok/s
**Amortize the 35ms across multiple tokens per commit.** This requires:
1. Speculative/lookahead decode (predict next N tokens, verify in batch)
2. OR: on-GPU argmax (skip CPU roundtrip entirely)
3. OR: reduce Metal scheduling overhead (Apple driver change — unlikely)

### Speculative Decode Results
- N-gram lookup: no improvement (code is unpredictable)
- Need: model-based speculation (draft model or self-speculation)
- The original test with draft model gave 21.9 tok/s — worse because running TWO models doubles the 35ms overhead
- Need: speculation that stays within ONE model / ONE command buffer
