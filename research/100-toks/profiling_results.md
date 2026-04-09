# Metal GPU Profiling Results — M4 16GB

## Test: 150 tokens decode at 64K context

### The Smoking Gun
```
GPU active time:  698 ms  (27%)
GPU idle time:   1887 ms  (73%)  ← CPU dispatch overhead
```

### Dispatch Pattern
- Total encoder dispatches: 98
- ~0.65 dispatches per token per layer
- Median gap between dispatches: 3.6 ms
- 27% of dispatches are <100µs (overhead-dominated)

### Duration Distribution
| Bucket | Count | % of dispatches | Time | % of GPU time |
|--------|-------|----------------|------|---------------|
| <100µs (tiny) | 27 | 27% | 0.6 ms | 0.1% |
| 100µs-1ms | 24 | 24% | 10.5 ms | 1.5% |
| 1-10ms | 39 | 39% | 165.8 ms | 23.8% |
| >10ms (large) | 8 | 8% | 521.0 ms | 74.7% |

### Key Insight
The GPU does its work fast. 8 large dispatches (>10ms) account for 75% of actual
GPU compute. But between dispatches, the CPU takes 3.6ms median to prepare the
next kernel. This is pure dispatch overhead.

### Path to 100 tok/s
If we eliminate 73% idle time → GPU runs at near 100% utilization → ~100 tok/s
Even halving the idle time → ~50 tok/s
