# MLX: How It Avoids the 35ms Problem

**Sources**:
- https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- https://deepwiki.com/ml-explore/mlx/3.1-computation-graph-model
- https://machinelearning.apple.com/research/exploring-llms-mlx-m5

## MLX's Architecture

MLX uses lazy evaluation: operations are recorded into a compute graph and only
executed when a value is actually needed (e.g., when you call `mx.eval()` or access
the numpy array).

### Key Difference from llama.cpp

**llama.cpp**: Build graph -> encode to command buffer(s) -> commit -> waitUntilCompleted -> read result -> repeat
**MLX**: Build graph lazily -> fuse operations -> encode minimal command buffers -> eval

### Why MLX is Faster for Decode

1. **Operation fusion**: MLX fuses multiple operations into single Metal kernels.
   Where llama.cpp might dispatch 10 separate compute commands for norm+rotate+scale+
   quantize+matmul, MLX can fuse these into fewer kernels. Fewer dispatches = less
   scheduling overhead.

2. **Unified memory zero-copy**: MLX tensors live in unified memory natively. No
   buffer mapping or mmap indirection. The GPU reads directly from the same physical
   memory as the CPU.

3. **Batched evaluation**: `mx.eval()` collects the entire pending graph and submits
   it in one shot. This means one command buffer commit per eval, not per-op.

4. **No mmap overhead**: MLX loads model weights into MTLBuffer objects directly
   (not via mmap). This means no page fault overhead during GPU access. The weights
   are pinned in GPU-accessible memory from the start.

### Performance Numbers

- MLX: ~230 tok/s sustained decode (on M2 Ultra, smaller models)
- MLX: 5-7ms latency, ~12ms P99
- llama.cpp: typically 20-30% slower than MLX on same hardware for decode

### The Catch

MLX doesn't support IQ3_S quantization or TurboQuant KV cache. The smallest Gemma 4
26B in MLX is 4-bit (~15GB), which doesn't fit in 16GB without swap thrashing.

## Takeaway

MLX proves that Metal CAN achieve low per-token decode latency. The problem is not
inherent to Metal -- it's specific to how llama.cpp submits command buffers with
synchronous waits and mmap'd model data.
