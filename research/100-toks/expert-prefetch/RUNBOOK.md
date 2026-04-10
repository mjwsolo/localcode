# Expert-Aware Prefetch System for MoE Models

## Hypothesis
Per-token decode latency on Gemma 4 26B (IQ3_S, M4 16GB) is wildly inconsistent
(5ms to 145ms per token, mean 35ms). When consecutive tokens activate the SAME
MoE experts, the expert weight pages are warm in the page cache (fast). When
different experts are activated, cold page faults cause stalls (slow).

## How It Works

### Architecture
```
Token N decode completes
    |
    v
[Extract expert indices from ffn_moe_topk-* tensors in graph]
    |
    v
[Issue posix_madvise(MADV_WILLNEED) on expert weight pages]
    |
    v
[Kernel starts async page-in of expert weights]
    |
    v
Token N+1 decode starts
    |
    v
[Expert weight pages already resident -> no stall]
```

### Why madvise works on Apple Silicon Metal
- Expert weights are mmap'd from the GGUF file
- Metal uses `newBufferWithBytesNoCopy` to wrap the mmap region
- The mmap pages ARE the Metal GPU pages (unified memory)
- `posix_madvise(MADV_WILLNEED)` triggers async page-in by the kernel
- By the time Metal reads the weights, pages are already resident

## Usage

### Step 1: Profile Expert Activations (Log Only)
```bash
# Set LLAMA_EXPERT_LOG to enable CSV logging without prefetch
LLAMA_EXPERT_LOG=/tmp/expert_log.csv \
  ./build/bin/llama-server \
    --model ~/.gem/models/gemma-4-26b-it-iq3_s-imat.gguf \
    -ngl 999 --mmap -ctk q8_0 -ctv turbo4 \
    -fa on -c 32768 --threads 10 -b 2048 -ub 512 \
    -np 1 -fit off --cache-ram 0

# Generate ~50 tokens, then analyze:
python analyze_experts.py /tmp/expert_log.csv
```

### Step 2: Enable Prefetch
```bash
# Set LLAMA_EXPERT_PREFETCH=1 to enable madvise prefetching
LLAMA_EXPERT_PREFETCH=1 \
LLAMA_EXPERT_LOG=/tmp/expert_prefetch_log.csv \
  ./build/bin/llama-server ...
```

### Step 3: Compare Performance
```bash
# Without prefetch:
LLAMA_EXPERT_LOG=/tmp/no_prefetch.csv ./build/bin/llama-server ...
# Generate 50 tokens

# With prefetch:
LLAMA_EXPERT_PREFETCH=1 LLAMA_EXPERT_LOG=/tmp/with_prefetch.csv ./build/bin/llama-server ...
# Generate 50 tokens

# Compare:
python analyze_experts.py /tmp/no_prefetch.csv
python analyze_experts.py /tmp/with_prefetch.csv
```

## Environment Variables
| Variable | Description |
|----------|-------------|
| `LLAMA_EXPERT_PREFETCH` | Set to `1` to enable madvise prefetching |
| `LLAMA_EXPERT_LOG` | Path to CSV log file for expert activation analysis |

## CSV Log Format
```
token_id,time_ms,toks_per_sec,layer_id,experts,overlap_count,overlap_ratio,page_faults
0,35.2,28.4,3,12;45;67;89;102;110;3;55,0,0.000,0
1,18.1,55.2,3,12;45;67;89;102;110;3;55,8,1.000,0
```

## Files Modified
- `src/llama-expert-prefetch.h` - Expert prefetch system header
- `src/llama-expert-prefetch.cpp` - Implementation (madvise + logging)
- `src/llama-context.h` - Added expert_prefetch member
- `src/llama-context.cpp` - Integrated into decode path + perf output
- `src/CMakeLists.txt` - Added new source file

## Key Design Decisions

1. **Between-token prefetch, not mid-graph**: The eval callback approach would
   synchronize after every graph node, destroying GPU pipelining. Instead, we
   extract expert indices AFTER graph compute completes and prefetch for the
   NEXT token.

2. **Same-expert bet**: We prefetch the same experts that were just used,
   betting on temporal locality. For code generation, consecutive tokens often
   activate similar experts.

3. **No performance cost**: The madvise call is non-blocking. The tensor readback
   (`ggml_backend_tensor_get`) is small (8 int32s per layer) and happens after
   the GPU sync that already occurs at the end of decode.

4. **Profiling-first**: The CSV logging mode works without prefetching enabled,
   allowing correlation analysis before committing to the approach.
