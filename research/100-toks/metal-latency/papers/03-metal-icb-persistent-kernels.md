# Metal Indirect Command Buffers and Persistent Kernels

## Sources
- [Encoding Indirect Command Buffers on GPU - Apple Docs](https://developer.apple.com/documentation/Metal/encoding-indirect-command-buffers-on-the-gpu)
- [Modern Rendering with Metal - WWDC19](https://developer.apple.com/videos/play/wwdc2019/601/)
- [Metal Indirect Command Encoding - Apple Docs](https://developer.apple.com/documentation/metal/indirect-command-encoding)
- [MTLIndirectCommandBuffer - Apple Docs](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer)

## Indirect Command Buffers (ICBs)

Metal 3 added compute dispatch support to ICBs. Key capabilities:

### What ICBs CAN do:
- Encode compute dispatches without CPU intervention
- Change kernels/pipelines between dispatches
- Change bound resources between dispatches
- Control serialization vs concurrent execution of dispatches
- Be built once and reused many times
- Be encoded by GPU compute kernels (GPU-driven pipelines)

### What ICBs CANNOT do:
- **Loop**: ICBs are a fixed sequence of commands. There is no conditional
  branching or looping construct. You cannot encode "run this N times until
  condition X."
- **Dynamic dispatch count**: The number of commands is fixed at creation time.
  You can skip commands (set them to NOP) but not add new ones.
- **Read back results**: A dispatch in an ICB cannot read the output of a
  previous dispatch in the same ICB to make decisions about what to dispatch next.
  The dispatches are pre-encoded.

### Bottom line for our use case:
ICBs are designed for GPU-driven RENDERING pipelines (frustum culling -> draw calls)
where the GPU decides WHICH pre-built commands to execute, not for iterative
compute loops where each iteration depends on the previous one's output.

**ICBs cannot implement an autoregressive decode loop.**

## Persistent Kernels

Metal does NOT support persistent GPU threads in the CUDA sense. Specifically:
- No `__global__ void persistent_kernel() { while(true) { ... } }` equivalent
- Metal compute kernels must terminate
- No guarantee of thread residency across dispatch boundaries

### Workaround: Long-running single dispatch
You CAN write a single Metal compute kernel that does a lot of work internally,
including loops. But:
- The entire model forward pass involves hundreds of different kernel types
  (matmul, softmax, RoPE, dequant, etc.) each needing different thread configurations
- A single "mega kernel" that does everything is impractical due to:
  - Register pressure (Metal has limited registers per thread)
  - Threadgroup size constraints
  - Different operations need different parallelization strategies
  - The ggml graph has ~500+ nodes per decode step

## What About Metal Argument Buffers?

Argument buffers allow passing complex data structures to shaders, including
pointers to other buffers. In theory, you could:

1. Put the model weights, KV cache, and token buffer in an argument buffer
2. Write a kernel that reads the current token, performs a simplified forward pass,
   writes the argmax result back to the token buffer, and loops

But this is essentially "rewrite the entire inference engine as a single Metal shader"
which is:
- Massive engineering effort (months)
- Would lose all ggml optimizations
- Would be fragile and model-specific
- Register/threadgroup limits make this impractical for large models

## Conclusion

ICBs and persistent kernels are NOT viable paths for eliminating per-token
command buffer overhead. The autoregressive dependency (each token depends on
the previous one's output) fundamentally requires either:
1. Multiple command buffer commits (current approach)
2. A multi-step graph that chains N forward passes in one commit
3. Some form of speculation that allows batching
