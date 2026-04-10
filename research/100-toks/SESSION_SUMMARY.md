# Speed Research Session Summary — 2026-04-09/10

## 12 Experiments Run

| # | Experiment | Result | Insight |
|---|-----------|--------|---------|
| 1 | Single command buffer | -11.5% | Serializes CPU encoding |
| 2 | Spin-wait sync | Same | 35ms is not the wait mechanism |
| 3 | N-gram speculation | -8.5% | Code is unpredictable |
| 4 | Expert deferral top-6 | -5.8% | Bottleneck isn't expert count |
| 5 | madvise pre-fault | -14.4% | Causes swap pressure on 16GB |
| 6 | Embedding on GPU (1 split) | -12.2% | Memory pressure worse |
| 7 | --no-mmap | -10.2% | malloc worse than mmap on 16GB |
| 8 | K=2 graph chain | -58.4% | Sequential GEMV reads weights 2x |
| 9 | K=4 server speculation | -65% | Verification rejects all drafts |
| 10 | CPU MoE | -96% | 62 graph splits kills performance |
| 11 | 2 parallel sequences | +38% combined | Proves batching amortizes overhead |
| 12 | Direct C API (no HTTP) | Same | Zero server overhead |

## What We Proved

1. **Metal scheduling is 0.41ms** (adversarial benchmark with 9 tests)
2. **35ms = reading ~1GB weights from DRAM** at 28% bandwidth utilization
3. **Batched prompt eval: 146 tok/s** — hardware CAN do it
4. **No server overhead** — direct API same speed as HTTP
5. **64K context works** — just `-c 65536`
6. **Weights don't stay warm** between consecutive tokens on 16GB

## Root Cause (Definitive)

On 16GB with 10.4GB model:
- Each token reads ~1GB of MoE expert weights
- At 34 GB/s effective bandwidth = 35ms per token
- No engineering trick reduces the bytes to read or increases the bandwidth
- The ONLY proven speedup: batching (reads weights once for N tokens)

## What Would Actually Work

1. **More RAM** (24GB+ = model fully resident = ~50+ tok/s)
2. **Smaller model** (IQ2_S 26B ~7GB = ~25ms = ~40 tok/s)
3. **True multi-token prediction** (MARS-style, requires model retraining)
4. **Windowed n-gram speculation** (54.9% acceptance, projected 77 tok/s — needs implementation work)

## Assets Created

- Bridge argmax Metal kernel (compiled)
- K=2 graph-chained forward pass (compiles, runs, produces tokens)
- Windowed n-gram speculative decoder (C++ implementation, needs wiring)
- Direct API ctypes bridge
- Adversarial Metal benchmark (9 tests, runnable)
- Complete profiling data (Xcode Instruments trace)
- 4 cross-domain analogy analyses
- First-principles innovation document with moonshot spec

## Branches

- `feature/turboquant-kv-cache` — WORKING (don't touch)
- `experiment/gpu-autonomous-decode` — K=2 graph chain + bridge kernel
- `experiment/better-speculation` — windowed n-gram decoder
- `experiment/speed-dispatch-optimization` — dispatch experiments
- `experiment/expert-deferral-top6` — top-6 experiment
- `experiment/madvise-prefault` — madvise experiment
