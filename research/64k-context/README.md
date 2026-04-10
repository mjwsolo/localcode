# 64K Context Research

Goal: Push context window from 32K to 64K tokens on 16GB Apple Silicon.

## Current State
- 32K context in 355 MiB KV cache (TurboQuant q8_0-K + turbo4-V)
- 3.8x KV compression via asymmetric quantization
- Server: llama-server with `-ctk q8_0 -ctv turbo4 -fa on -c 32768`

## Hypothesis
- 64K = ~710 MiB KV cache — should fit within 14GB Metal budget
- TurboQuant compression makes this feasible where raw KV would OOM
- Main risk: quality degradation at 64K with turbo4 quantization

## Test Plan
1. Benchmark quality at 32K vs 64K (needle-in-haystack test)
2. Memory profiling at 64K (does it fit? swap pressure?)
3. Prompt eval speed at 64K (how much slower?)
4. Real-world test: large codebase refactor with full context

## Key Files
- `benchmark_64k.py` — quality + memory benchmarks
- `server_config_64k.sh` — llama-server launch with 64K config
