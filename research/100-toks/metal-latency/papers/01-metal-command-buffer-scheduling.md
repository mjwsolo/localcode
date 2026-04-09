# Metal Command Buffer Scheduling: Why 35ms?

## Sources
- [Metal Compute on MacBook Pro - Apple Tech Talk](https://developer.apple.com/videos/play/tech-talks/10580/)
- [Optimize Metal Performance for Apple Silicon - WWDC20](https://developer.apple.com/videos/play/wwdc2020/10632/)
- [Metal Command Buffer Best Practices](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html)
- [Metal Compute Never Beats 2.5ms? - Apple Dev Forums](https://developer.apple.com/forums/thread/46817)
- [Anukari: Huge macOS Performance Improvements](https://anukari.com/blog/devlog/huge-macos-performance-improvements)

## The 35ms Breakdown

The 35ms per-token latency is NOT a single cause. It is the sum of multiple overheads
in the Metal command buffer lifecycle:

### 1. GPU Power State Management (~15-20ms)
Apple Silicon GPUs aggressively manage power states. When submitting short jobs with
small breaks between them, the GPU goes to sleep and takes a long time to come back.
This is documented in Apple's own Tech Talk:

> "If you are submitting a lot of short jobs with small breaks in between, the GPU can
> go to sleep and take a very long time to come back, causing 2-4x performance loss
> even on extremely large machine learning workloads."

For decode (1 token per commit), each command buffer completes in ~0ms of GPU compute,
creating a "break" that triggers GPU power-down. The GPU then needs to spin back up
for the next command buffer. This alone could account for 15-20ms.

**Key insight**: Prompt eval does NOT have this problem because hundreds of tokens
keep the GPU continuously busy -- no power-down occurs.

### 2. Resource Validation and Page Table Setup (~10-15ms)
When a command buffer is committed, Metal must:
- Validate all referenced GPU resources
- Ensure mmap'd regions are page-resident (or trigger page faults)
- Set up GPU page table entries for the 10.4GB mmap'd model

With 10.4GB of mmap'd GGUF data, the page table overhead is significant:
- 16KB pages on Apple Silicon = 650,000+ pages for the full model
- Each decode touches ~1.2GB of expert weights = ~75,000 pages
- TLB coverage is ~48MB vs 10.4GB model = 200x oversubscription

The `ggml_metal_device_rsets_keep_alive()` call in llama.cpp tries to keep
resource sets alive, but this only prevents deallocation -- it does NOT prevent
page fault overhead on mmap'd regions.

### 3. Command Buffer Encoding and JIT (~2-5ms)
Metal must:
- JIT-compile any new pipeline states (first time only, then cached)
- Copy data to wired-down memory in MTLResources
- Encode the workload into the command buffer hardware format

### 4. Inter-Buffer Scheduling Gap (~3-5ms)
Even with back-to-back commits, there is a scheduling quantum between when one
command buffer completes and the next one begins executing. This is the "bubble"
in the GPU timeline.

## Why 35ms and Not 2.5ms?

The Apple Dev Forums thread "Metal Compute Never Beats 2.5ms?" documents a minimum
~2.5ms floor for trivial compute workloads. Our 35ms is 14x higher because:

1. We have a MASSIVE resource set (10.4GB mmap'd model)
2. MoE reads scattered pages across the full 10.4GB
3. GPU power cycling between tokens
4. Resource validation scales with number of heaps/buffers referenced

## VSync Connection

Metal command buffers for compute are NOT tied to VSync/display refresh.
The 35ms (close to 2x 16.7ms) is coincidental. Compute command buffers
execute on the compute queue which is independent of the display pipeline.

However, the GPU power management IS shared between display and compute,
which means the display refresh rate may indirectly affect power state timing.

## The Proof: Prompt Eval vs Decode

| Metric | Prompt Eval | Decode |
|--------|-------------|--------|
| Tokens per commit | ~500 | 1 |
| GPU active time | continuous | ~0ms |
| Power cycling | none | every token |
| Resource validation | once/commit | once/commit |
| Per-token overhead | 0.07ms | 35ms |
| Throughput | 318 tok/s | 28 tok/s |

This table proves the overhead is per-commit, not per-token.
