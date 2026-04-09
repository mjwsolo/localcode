# 100 tok/s Action Plan

## Confirmed: Dispatch Overhead is 73% of Wall Time

GPU active: 698ms (27%), GPU idle: 1887ms (73%)
98 command buffer dispatches for 150 tokens, median 3.6ms gap between them.

## Attack Vector: Reduce Command Buffer Count + Overlap CPU/GPU

### Phase 1: Quick wins (this week)

#### 1A. Expert Deferral — top-6 instead of top-8
- Reduce bytes read per token by 25% (1.2GB → 0.9GB)
- Less work per command buffer = faster GPU completion
- Modify router threshold in llama.cpp MoE layer
- **Expected: +25% speed → ~35 tok/s**
- **Effort: 2-3 days**
- **File: src/llama-model.cpp — MoE routing logic**

#### 1B. GGUF Tensor Reordering
- Group per-layer tensors contiguously in GGUF file
- Better mmap readahead, fewer TLB misses
- Write Python script to reorder GGUF tensors
- **Expected: +10-20% → ~38-42 tok/s**
- **Effort: 3-4 days**

#### 1C. mlock + madvise for hot weights
- Pin attention weights + frequently-used experts in memory
- Eliminate page faults during decode
- **Expected: +5-10%**
- **Effort: 1 day**

### Phase 2: Reduce dispatch overhead (weeks 2-3)

#### 2A. Double-buffered command encoding
- While GPU executes command buffer N, CPU prepares command buffer N+1
- Overlaps the 3.6ms CPU time with GPU execution
- Requires restructuring the encode_async loop
- **Expected: up to 2x if CPU prep fully overlapped → ~60-70 tok/s**
- **Effort: 1-2 weeks**
- **File: ggml-metal-context.m — ggml_metal_graph_compute**

#### 2B. Increase nodes per command buffer
- Currently: n_nodes_0 = MAX(64, 0.1*n_nodes)
- More nodes per buffer = fewer commits = fewer gaps
- Risk: larger command buffers may have longer GPU latency
- **Expected: +20-30%**
- **Effort: 1-2 days (just change a constant)**

### Phase 3: Custom Metal kernels (weeks 3-6)

#### 3A. Fused MoE kernel
- Single Metal shader: gate → select experts → parallel GEMV → weighted sum
- Eliminates ~12 dispatches per MoE layer per token
- 26 MoE layers × 12 saved dispatches = 312 fewer dispatches per token
- **Expected: +30-50% → potentially 80-100 tok/s**
- **Effort: 3-4 weeks of Metal shader development**

#### 3B. Fused attention kernel
- Single shader: RMSNorm → Q/K/V proj → RoPE → attention → output proj
- Eliminates ~7 dispatches per attention layer per token
- **Expected: +20-30%**
- **Effort: 2-3 weeks**

## Cumulative Projection

| Phase | Technique | Speed | Cumulative |
|-------|-----------|-------|------------|
| Baseline | — | 28 tok/s | 28 |
| 1A | Expert deferral | +25% | 35 |
| 1B | GGUF reorder | +15% | 40 |
| 2B | More nodes/buffer | +25% | 50 |
| 2A | Double-buffer | +40% | 70 |
| 3A | Fused MoE | +40% | 98 |

## Start With Phase 2B — It's a One-Line Change

Change `n_nodes_0 = MAX(64, 0.1*gf->n_nodes)` to a higher value.
This batches more ops per command buffer, reducing the number of commits.
Fastest way to validate if dispatch overhead is truly the bottleneck.
