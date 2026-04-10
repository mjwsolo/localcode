# 64K Context Hypotheses — Prioritized by (Impact x Feasibility)

## Current Baseline

- **Model**: Gemma 4 26B-A4B (IQ3_S, 10.4GB)
- **KV config**: q8_0-K + turbo4-V = 355 MiB at 32K context
- **Architecture**: 30 layers total
  - 26 sliding window layers (1024-token window, 256 head_dim, 4 KV heads)
  - 4 global layers (full context, 512 head_dim, 4 KV heads)
- **GPU budget**: ~14GB via sysctl, model uses ~10.4GB, leaving ~3.6GB for KV + overhead
- **Target**: 64K context within existing memory budget

## KV Cache Math (FP16 Reference)

Per token per layer (one of K or V):
- Sliding layer: 4 KV heads x 256 dim x 2 bytes = 2,048 bytes
- Global layer: 4 KV heads x 512 dim x 2 bytes = 4,096 bytes

At 32K context, FP16:
- 26 sliding layers: 1024 tokens x 2048 bytes x 2 (K+V) x 26 = 109 MiB
- 4 global layers: 32768 tokens x 4096 bytes x 2 (K+V) x 4 = 1,024 MiB
- **Total FP16**: ~1,133 MiB

With q8_0-K + turbo4-V (current):
- K at q8_0 (~50% of FP16): ~567 MiB -> ~283 MiB
- V at turbo4 (3.8x = ~26% of FP16): ~567 MiB -> ~149 MiB
- **Total current**: ~355 MiB (matches observed) — but note: sliding layers
  only store 1024 tokens, so global layers dominate

At 64K, the sliding layers DON'T grow (fixed 1024 window). Only global layers double:
- 26 sliding: same ~28 MiB (with current compression)
- 4 global at 64K: ~655 MiB (with current compression)
- **Total at 64K with current config**: ~683 MiB

This is **already feasible** within the ~3.6 GB headroom. The question is
whether llama.cpp actually implements the sliding window correctly for Gemma 4
or allocates full context for all layers.

---

## Hypothesis 1: Verify Sliding Window KV Allocation (URGENT)

**Priority: CRITICAL — investigate first**

**What**: Check whether llama.cpp allocates KV cache only for the 1024-token
window on the 26 sliding layers, or wastefully allocates full context for all 30 layers.

**Expected gain**: If llama.cpp is allocating full context for all layers,
fixing this alone could enable 64K:
- Wrong (all 30 layers at 32K): 30 layers worth of full-context KV
- Right (26 sliding at 1024 + 4 global at 64K): only 4 layers scale with context

**Risk**: Low. This is verification, not a code change.

**Effort**: 1-2 hours to inspect llama.cpp KV allocation code.

**Action**: Grep llama.cpp source for sliding_window, n_ctx_per_layer, or
per-layer KV allocation. Check if Gemma 4's `sliding_window_pattern: 6`
config is respected in KV sizing.

---

## Hypothesis 2: Switch V Cache to turbo3 (-ctv turbo3)

**Priority: 1 (Highest impact x feasibility)**

**What**: Change V cache from turbo4 (3.8x) to turbo3 (4.9x).

**Expected gain**:
- At 64K with q8_0-K + turbo3-V (global layers only scaling):
  - K: ~328 MiB (global layers at 64K + sliding at 1024)
  - V: ~255 MiB (global at 64K/4.9x + sliding at 1024/4.9x)
  - Total: ~583 MiB
- Saves ~100 MiB vs turbo4 at 64K

**Quality**: turbo3 shows negligible PPL impact at 13B+ models. The TheTom
fork reports PPL 6.20 vs 6.19 baseline on Qwen 3.5-35B.

**Risk**: Low. turbo3 is well-tested in the ecosystem. Our model is already
on TheTom's TurboQuant fork.

**Effort**: 30 minutes. Change config default and test.

---

## Hypothesis 3: Asymmetric turbo3-K + turbo2-V

**Priority: 2 (High impact, medium feasibility)**

**What**: Compress both K and V more aggressively:
- K: turbo3 (4.9x) instead of q8_0 (2x)
- V: turbo2 (6.4x) instead of turbo4 (3.8x)

**Expected gain**:
- At 64K (global layers dominant):
  - K: ~133 MiB (global at 64K/4.9x)
  - V: ~102 MiB (global at 64K/6.4x)
  - Total: ~235 MiB — massive headroom for even 96K+

**Quality**: turbo3-K is safe. turbo2-V adds +6.48% PPL which stacked on
IQ3_S weights is the main risk. Need to test actual output quality, not just PPL.

**Risk**: Medium. turbo2 quality on already-quantized model is unknown.
Could cause noticeable generation degradation on complex code tasks.

**Effort**: 2-4 hours. Pull turbo2 support from TheTom's fork, test quality.

---

## Hypothesis 4: KIVI-style Per-Channel K + Per-Token V at 2-bit

**Priority: 3 (High impact, medium-hard feasibility)**

**What**: Implement KIVI's asymmetric 2-bit strategy:
- Keys: 2-bit per-channel quantization (exploits K's column-wise distribution)
- Values: 2-bit per-token quantization (exploits V's row-wise distribution)

**Expected gain**:
- 8x compression over FP16 for both K and V
- At 64K: ~80 MiB for global layers — enables 128K+ easily

**Quality**: KIVI maintains "almost the same quality" on Llama/Mistral at
similar sizes. But untested on IQ3_S quantized Gemma 4 MoE.

**Risk**: Medium-high. Per-channel K quantization needs full channel stats,
which means a calibration pass or running statistics. Different from
TurboQuant's per-block approach.

**Effort**: 1-2 weeks. New GGML quantization type, Metal kernels, calibration.

---

## Hypothesis 5: Layer-Discriminative KV Budgets for Global Layers

**Priority: 4 (Medium impact, medium feasibility)**

**What**: Profile the 4 global attention layers to determine if they all need
full 64K context, or if some can use shorter effective windows.
Based on SqueezeAttention and MiniKV insights.

**Expected gain**: If 2 of 4 global layers can work with 16K effective context:
- Saves ~50% of global layer KV for those layers
- Could save ~150 MiB at 64K

**Quality**: Depends on which layers are retrieval-critical. Needs profiling
with coding tasks to ensure code recall isn't degraded.

**Risk**: Medium. Profiling may show all 4 global layers are critical,
yielding no savings.

**Effort**: 3-5 days. Profile layers, implement per-layer context limits.

---

## Hypothesis 6: DuoAttention-style Head Profiling on Global Layers

**Priority: 5 (Medium impact, medium feasibility)**

**What**: Within the 4 global layers, profile which of the 4 KV heads are
"retrieval" vs "streaming" heads. Streaming heads only need recent tokens +
sinks, even in global layers.

**Expected gain**: If 2 of 4 KV heads per global layer are streaming:
- Those heads only need ~4K token window instead of 64K
- Saves ~50% of KV for those heads = ~25% overall global savings

**Quality**: DuoAttention outperforms H2O/StreamingLLM at same budget.
The profiling algorithm is well-validated.

**Risk**: Medium. GQA with only 4 KV heads gives less granularity than
MHA models where DuoAttention was originally tested.

**Effort**: 1 week. Head profiling, custom per-head KV allocation.

---

## Hypothesis 7: Sparse Attention with Landmark Selection (ShadowKV-style)

**Priority: 6 (High impact, hard feasibility)**

**What**: For global layers at 64K, don't attend to all tokens. Use landmark
summaries to select top ~2% of tokens per decode step, attend only to those.

**Expected gain**: 
- Active KV per decode step: ~1,300 tokens instead of 64K
- Massive compute savings (attention is O(n), not O(64K))
- Memory for landmarks is tiny (~1-2 MiB)
- Full KV stored compressed, fetched on demand

**Quality**: ShadowKV shows minimal quality loss with 1.56% selection on
standard benchmarks including coding tasks.

**Risk**: High. Requires fundamental changes to attention computation.
Sparse attention may miss relevant code context that landmark selection
doesn't predict well.

**Effort**: 2-4 weeks. Custom attention kernel, landmark computation,
sparse KV fetch logic.

---

## Hypothesis 8: NVMe-backed KV Cache (KVSwap-style)

**Priority: 7 (Enables 128K+, medium feasibility)**

**What**: Store full KV cache on NVMe SSD, keep metadata in RAM, fetch
needed KV pairs during decode with async I/O.

**Expected gain**: Context limited only by SSD capacity, not RAM.
M4 NVMe: ~7 GB/s read. Fetching 50 MiB of selected KV per step: ~7ms.

**Quality**: No loss if all needed tokens are fetched.

**Risk**: Adds 5-10ms latency per decode step (at 27 tok/s that's 37ms
baseline, so +7ms = ~20% slowdown). Write amplification on prefill.

**Effort**: 1-2 weeks. Async I/O layer, metadata index, integration with
llama.cpp KV management.

---

## Hypothesis 9: KVTC for Conversation Persistence

**Priority: 8 (Not for live 64K, but valuable for UX)**

**What**: Use KVTC's 20x compression to save/load conversation KV caches
to disk between sessions. Resume conversations without re-prefilling.

**Expected gain**: Not a live context extension, but eliminates re-prefill
cost for returning conversations. 64K context saved in ~35 MiB on disk.

**Quality**: KVTC maintains quality at 20x on reasoning benchmarks.

**Risk**: Low (offline operation, can always re-prefill if quality is bad).

**Effort**: 1-2 weeks. PCA calibration, entropy coding, save/load integration.

---

## Recommended Execution Order

### Phase 1: Quick Wins (This Week)
1. **Hypothesis 1**: Verify sliding window allocation — if wrong, fixing it alone may enable 64K
2. **Hypothesis 2**: Test turbo3-V — 30 minutes, low risk, immediate savings

### Phase 2: Push Further (Next Week)
3. **Hypothesis 3**: Test turbo3-K + turbo2-V — measure quality on real coding tasks
4. **Hypothesis 5**: Profile global layer importance — identify if any can be compressed more

### Phase 3: Advanced (2-4 Weeks)
5. **Hypothesis 4**: KIVI-style 2-bit — implement proper per-channel/per-token quantization
6. **Hypothesis 6**: Head profiling — DuoAttention analysis on Gemma 4's GQA heads

### Phase 4: Frontier (Month+)
7. **Hypothesis 7**: Sparse attention — ShadowKV-style landmark selection
8. **Hypothesis 8**: NVMe-backed KV — enables 128K+ for truly long contexts
9. **Hypothesis 9**: KVTC persistence — conversation save/restore

---

## Key Insight: Gemma 4's Architecture Is Already Optimized

The most important finding from this research is that Gemma 4 26B-A4B's
hybrid sliding/global architecture already implements the core insight from
DuoAttention and StreamingLLM:

- **26 of 30 layers** use 1024-token sliding windows (constant memory)
- **Only 4 layers** use full-context global attention

This means 64K context scaling only affects 4 layers. If llama.cpp
properly implements the sliding window (Hypothesis 1), the jump from
32K to 64K costs only ~330 MiB of additional KV cache for the global
layers — well within our memory budget even with current turbo4 compression.

**The 64K goal may already be achievable with zero new compression work.**
The first action item is to verify the sliding window implementation and
test 64K with `-c 65536` using current settings.
