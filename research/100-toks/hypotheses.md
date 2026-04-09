# Hypotheses for Reaching 100 tok/s on M4 16GB

## Current State Analysis

| Metric | Value |
|--------|-------|
| Current decode speed | 27-28 tok/s |
| Target | 100 tok/s |
| Gap | 3.6x improvement needed |
| Memory bandwidth (M4) | ~120 GB/s |
| Active weights per token | ~1.2GB (3.8B params at IQ3_S ~3.5 bpw) |
| Theoretical max (BW-limited) | ~100 tok/s |
| Current BW utilization | ~28% |
| Overhead sources | Metal dispatch, graph splits, kernel barriers, TLB misses, cache misses |

## The Fundamental Equation

```
tok/s = memory_bandwidth / bytes_read_per_token
100 tok/s = 120 GB/s / 1.2 GB  (requires ~100% BW utilization)
```

There are exactly two paths to 100 tok/s:
1. **Increase effective bandwidth utilization** from 28% toward 100%
2. **Reduce bytes read per token** below 1.2 GB

Most realistic: combine both approaches.

---

## Hypothesis Rankings (Impact x Feasibility)

### TIER 1: Highest Priority (Expected 2-4x combined gain)

---

#### H1: Reduce Metal Kernel Dispatch Overhead via Mega-Kernels
**Score: 9/10**

**What**: Fuse multiple small Metal kernels (dequant + matmul + activation + residual) into single "mega-kernels" that process an entire MoE layer in one dispatch. Currently, each expert likely triggers separate kernel dispatches with synchronization barriers between them.

**Expected gain**: 1.5-2x. Multiple papers and ZINC benchmarks show ~28% utilization across all engines, suggesting Metal dispatch overhead consumes ~70% of wall time. Reducing kernel count from ~50+ per layer to ~5-10 could recover significant dispatch overhead.

**Evidence**: 
- ZINC (hand-tuned Metal) achieves same ~28% utilization as llama.cpp -- suggests Metal dispatch is the bottleneck, not framework inefficiency
- llama.cpp uses 2 graph splits for our model, but each graph still contains many individual kernel dispatches
- Paper 09 shows MoE SM utilization is 28-34% even on NVIDIA, suggesting this is partly inherent to MoE GEMV arithmetic intensity

**Risk**: Medium. Metal has limits on kernel complexity. Fused kernels may not improve bandwidth (the bottleneck might be DRAM controller, not dispatch). Need to profile Metal GPU traces first.

**Effort**: 2-3 weeks. Requires writing custom Metal shaders and modifying ggml-metal.m.

**First step**: Profile with Xcode Metal System Trace to identify exactly how much time is dispatch overhead vs actual memory/compute.

---

#### H2: Expert Weight Prefetching via Cross-Layer Prediction
**Score: 8.5/10**

**What**: Predict which experts will be needed in layer N+1 while computing layer N, then issue madvise(MADV_WILLNEED) or manual prefetch to warm those expert weights into cache/DRAM page tables.

**Expected gain**: 1.3-1.8x. If we can overlap 50-70% of expert weight reads with current-layer computation, effective throughput increases proportionally.

**Evidence**:
- Pre-attention expert prediction achieves 93-97% accuracy (paper 02)
- Cross-layer gate prediction is even simpler and zero-overhead (paper 07)
- Residual-based prediction requires no training (paper 14)
- Score-based prefetching outperforms binary prediction (paper 08)

**Implementation plan**:
1. Profile Gemma 4's cross-layer expert correlation (which layer N experts predict layer N+1?)
2. If correlation >85%: implement simple cross-layer prediction
3. If correlation <85%: train lightweight linear predictors (2 small matrices per layer)
4. Add madvise(MADV_WILLNEED) calls for predicted expert weight pages between layers
5. Consider manual prefetch via dummy reads in a background thread

**Risk**: Low-medium. Apple Silicon unified memory means "prefetch" is really "warm the page table / L2 cache line." The benefit depends on whether our expert reads are hitting cold pages or just saturating DRAM bandwidth. If it's the latter, prefetching won't help.

**Effort**: 1-2 weeks for profiling + implementation.

**First step**: Write a simple script that traces which experts are activated per layer per token and measures cross-layer correlation.

---

#### H3: GGUF Expert Tensor Reordering for Spatial Locality
**Score: 8/10**

**What**: Reorder tensors in the GGUF file so that each layer's attention weights + hot expert weights are contiguous in the file. Currently, GGUF stores tensors in alphabetical order by name, which interleaves expert tensors with attention tensors across the entire 10.4GB file.

This is the ROOT CAUSE of the mmap range bloat issue we already patched. But the patch only fixed OOM -- it didn't optimize the access pattern.

**Expected gain**: 1.2-1.5x. Better spatial locality means:
- Sequential reads instead of scattered reads across 10.4GB
- Better OS-level readahead (mmap pages fetched sequentially are prefetched by the kernel)
- Reduced TLB misses (fewer unique 16KB pages touched per token)
- Better utilization of Apple Silicon's SLC (System Level Cache)

**Evidence**:
- Our own mmap range bloat bug proves tensors are scattered
- Paper 13 (FlashMoE) shows expert locality matters for cache performance
- KTransformers uses AMX tiling-aware memory layout for the same reason
- TLB coverage: 48MB L2 TLB vs 10.4GB model = 200x oversubscription

**Implementation plan**:
1. Write a GGUF reordering tool that groups tensors by layer, then by type within layer:
   - Layer N attention weights (always-active, ~150MB total across all layers)
   - Layer N gate/router weights (tiny)
   - Layer N hot experts (top-16 by frequency, grouped together)
   - Layer N cold experts (remaining 112 experts)
2. Rebuild the model with this layout
3. Measure decode speed before/after

**Risk**: Low. Worst case: no improvement. GGUF format supports arbitrary tensor order. llama.cpp loads by tensor name, not position.

**Effort**: 1 week. Mostly a Python script to reorder GGUF tensors + profiling.

**First step**: Analyze current GGUF tensor layout to quantify scatter.

---

#### H4: Reduce Active Expert Count During Decode (Expert Deferral)
**Score: 7.5/10**

**What**: During decode, not all 8 active experts contribute equally. If we can identify and skip the 2-3 least important experts per token, we reduce bytes read from 1.2GB to 0.75-0.9GB per token, directly increasing throughput by 30-60%.

**Expected gain**: 1.3-1.6x (from reducing active experts from 8 to 5-6).

**Evidence**:
- KTransformers' Expert Deferral achieves <0.5% accuracy drop (paper 05)
- Router softmax scores indicate expert importance -- low-score experts contribute minimally
- For IQ3_S quantization, the noise from quantization may exceed the contribution of the weakest experts

**Implementation plan**:
1. Profile router score distributions: what's the typical range of expert scores?
2. Implement a dynamic threshold: skip experts whose score is below X% of the top expert's score
3. Measure quality impact on coding benchmarks at various thresholds
4. If quality holds at top-6 routing: modify llama-server to use top-6 during decode only (keep top-8 for prefill)

**Risk**: Medium. Quality degradation on complex reasoning tasks. May need per-layer thresholds since some layers are more sensitive.

**Effort**: 1 week. The router threshold is a small code change. Quality evaluation takes time.

**First step**: Log router scores during inference to see the score distribution.

---

### TIER 2: Medium Priority (Expected 1.2-1.5x each)

---

#### H5: Multi-Token Decode Batching
**Score: 7/10**

**What**: Instead of decoding one token at a time, speculatively decode 2-4 tokens in parallel by running the model on the last N generated tokens simultaneously. This amortizes expert weight reads across multiple tokens, increasing arithmetic intensity.

**Expected gain**: 1.5-2x at batch=2-4. Reading 1.2GB of weights but computing 2-4 tokens instead of 1 means 2-4x better FLOP/byte ratio. Even with verification overhead, net gain should be positive.

**Evidence**:
- This is the principle behind speculative decoding (which we tried and found slower with a draft model)
- BUT: self-speculative decoding (using the same model, just predicting N tokens greedily then verifying) avoids the draft model overhead
- Jacobi decoding / lookahead decoding does exactly this
- Paper 17 confirms speculative decode exploits the bandwidth-compute gap

**Key difference from what we tried**: We tried spec decode with a separate draft model (overhead of running two models). Self-speculation or Jacobi decoding uses the SAME model, just batches multiple positions.

**Risk**: High. Need to verify that llama.cpp's Metal backend efficiently handles batch=2-4 for a single sequence (not just multi-user batching). Gemma 4's MoE routing may differ per position, reducing the benefit.

**Effort**: 2-3 weeks. Requires modifying the decode loop and possibly Metal batch handling.

---

#### H6: mlock Hot Expert Weights + madvise Optimization  
**Score: 6.5/10**

**What**: Use mlock() to pin the always-active attention weights + top-16 most frequent experts in physical memory. Use madvise(MADV_SEQUENTIAL) for sequential layers and madvise(MADV_WILLNEED) for predicted experts.

**Expected gain**: 1.1-1.3x. Eliminates page fault stalls and ensures hot weights are never swapped.

**Evidence**:
- Paper 11: OS-level memory management has measurable impact on LLM inference
- macOS can and does page out mmap'd regions under memory pressure
- Our 16GB system with ~14GB GPU allocation leaves very little headroom

**Risk**: Low. May not help if pages are already resident. Could increase memory pressure on other system processes.

**Effort**: 2-3 days.

---

#### H7: Custom Metal Shader for MoE GEMV
**Score: 6/10**

**What**: Write a single Metal compute shader that performs the entire MoE forward pass for one layer: gate computation, expert selection, parallel GEMV across selected experts, weighted sum. Currently this is decomposed into many separate GGML operations.

**Expected gain**: 1.3-1.5x. Eliminates inter-kernel synchronization, reduces Metal command buffer overhead, enables better simdgroup utilization across experts.

**Evidence**:
- ZINC's hand-tuned Metal shaders achieve competitive performance
- ggml's generic Metal kernels don't specialize for MoE's unique access pattern
- Fusing gate + select + compute + reduce removes 5-8 kernel dispatches per layer

**Risk**: High. Writing correct, high-performance Metal shaders for quantized MoE is extremely difficult. Need to handle IQ3_S dequantization within the fused kernel.

**Effort**: 3-4 weeks.

---

### TIER 3: Exploratory (Potentially High Impact, High Uncertainty)

---

#### H8: Aggressive Expert Quantization (IQ2 / IQ1 for Cold Experts)
**Score: 5.5/10**

**What**: Keep top-16 hot experts at IQ3_S but quantize the remaining 112 cold experts to IQ2_S or IQ1_S. This reduces model size and bytes-per-token for cold expert reads.

**Expected gain**: 1.2-1.4x. If cold experts (rarely activated) are 50% smaller, average bytes-per-token drops ~20% for the MoE layers.

**Evidence**: HOBBIT (paper 03) shows mixed-precision expert loading with minimal quality loss. The insight is that cold experts are cold precisely because they're less important.

**Risk**: Medium-high. Quality impact needs careful measurement, especially since we're already at aggressive IQ3_S.

**Effort**: 1-2 weeks (re-quantize model, benchmark quality).

---

#### H9: Neural Engine Offload for Expert Computation
**Score: 4/10**

**What**: Apple's Neural Engine (ANE) is a separate compute unit with its own memory bandwidth. If we could offload some expert MoE computations to ANE while GPU handles attention, we'd get additional memory bandwidth.

**Expected gain**: Unknown, potentially 1.5-2x if ANE bandwidth adds to GPU bandwidth.

**Evidence**: 
- M5 paper shows ANE provides up to 4x speedup for matrix operations via MLX
- ANE has its own path to unified memory
- CoreML supports model splitting across GPU + ANE

**Risk**: Very high. ANE is designed for specific operation patterns (convolutions, quantized matmul). Getting arbitrary MoE expert GEMV to run efficiently on ANE is uncharted territory. No llama.cpp support.

**Effort**: 4+ weeks, likely requires CoreML integration.

---

#### H10: Lookahead/Jacobi Decoding with Expert Batching
**Score: 5/10**

**What**: Generate N candidate tokens in parallel (Jacobi iteration), verify them, accept correct prefix. The key win for MoE: if multiple candidate positions route to the SAME experts, those expert weights are read once but used N times.

**Expected gain**: 1.5-3x if expert overlap across positions is high. For Gemma 4 with 128 experts and top-8 routing, random overlap probability for any one expert between two tokens is ~12%. But in practice, expert selection is context-dependent and may cluster.

**Risk**: High. Implementation complexity is significant. Verification step adds overhead. Expert overlap needs profiling.

**Effort**: 3-4 weeks.

---

## Recommended Execution Order

### Phase 1: Profile and Measure (Week 1)
1. **Metal System Trace** - Xcode GPU profiling to identify dispatch overhead vs bandwidth utilization (validates H1)
2. **Expert activation profiling** - Log which experts activate per layer per token, measure cross-layer correlation (validates H2, H7)
3. **Router score distribution** - Log softmax scores to assess expert deferral feasibility (validates H4)
4. **GGUF tensor layout analysis** - Map tensor file offsets to identify scatter pattern (validates H3)

### Phase 2: Low-Hanging Fruit (Weeks 2-3)
5. **GGUF reordering** (H3) - Lowest risk, clear benefit path
6. **mlock + madvise** (H6) - Trivial to implement
7. **Expert deferral** (H4) - Small code change, measure quality impact
8. **Cross-layer expert prefetch** (H2) - If correlation is high from Phase 1

### Phase 3: High-Impact Engineering (Weeks 4-6)
9. **Metal mega-kernels** (H1) - Based on profiling data from Phase 1
10. **Multi-token decode** (H5) - If Metal profiling shows dispatch overhead is dominant

### Phase 4: Experimental (Weeks 7+)
11. **Mixed-precision experts** (H8)
12. **Jacobi decoding** (H10)
13. **ANE offload** (H9)

## Expected Cumulative Gains

Realistic scenario combining Tier 1 + Tier 2 techniques:

| Technique | Speedup | Cumulative tok/s |
|-----------|---------|-------------------|
| Baseline | 1.0x | 28 |
| GGUF reorder (H3) | 1.2x | 34 |
| Expert deferral top-6 (H4) | 1.25x | 42 |
| Expert prefetching (H2) | 1.4x | 59 |
| Metal mega-kernels (H1) | 1.5x | 89 |
| Multi-token decode (H5) | 1.2x | **107** |

This is optimistic but not unreasonable. Each multiplier is conservative for its category. The real question is whether they compound (multiply) or overlap (don't). Metal profiling in Phase 1 will tell us which gains are independent.

## Critical Open Questions

1. **Is the 28% utilization from dispatch overhead or DRAM controller saturation?**
   - If dispatch: H1 is king, massive gains possible
   - If DRAM: only reducing bytes (H4, H8) or batching (H5, H10) will help

2. **How strong is cross-layer expert correlation in Gemma 4?**
   - >90%: H2 is a slam dunk
   - <70%: need trained predictors or skip this approach

3. **How sensitive is IQ3_S Gemma 4 to expert deferral?**
   - At full precision, skipping 2/8 experts loses ~2% quality
   - At IQ3_S, the quantization noise may mask the deferral impact (good news)

4. **Does Metal's unified memory architecture inherently limit bandwidth utilization?**
   - If GPU and CPU both accessing DRAM simultaneously, effective bandwidth per consumer drops
   - Need to measure isolated GPU bandwidth vs theoretical
