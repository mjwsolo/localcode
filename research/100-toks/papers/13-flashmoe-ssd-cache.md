# FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement

- **Authors**: (Multiple authors)
- **Date**: January 2026
- **URL**: https://arxiv.org/abs/2601.17063

## Key Technique
ML-based cache replacement for MoE expert weights. Approximates Belady's optimal caching policy using a lightweight ML model that combines recency and frequency signals.

Key innovations:
- Frequency-gated admission filter: experts only enter cache on second miss
- Cold experts (seen once during prefill) never pollute the cache
- ML model predicts which experts to evict based on combined recency + frequency signals

## Results
- Up to 51% improvement in cache hit rate over LRU/LFU
- Up to 2.6x speedup over existing MoE inference systems
- Targets edge devices with limited RAM

## Relevance to 100 tok/s Goal
**MEDIUM** - Our entire model fits in memory via mmap, so we don't have the SSD offloading problem. However, the CACHE-LEVEL behavior matters:

On Apple Silicon, the GPU accesses unified memory through a cache hierarchy:
- L1 cache: tiny (~32KB per core)
- System-level cache (SLC): 16-32MB on M4
- DRAM: 120 GB/s

Expert weights that are "hot" (frequently accessed) could be kept in SLC if accessed in the right pattern. The ML-based prediction of hot vs cold experts could inform a software-managed expert weight layout that improves SLC hit rates.

## Implementation Difficulty
**LOW-MEDIUM** - Profile expert activation patterns on representative workloads, then:
1. Identify the top-N most frequently activated experts across all layers
2. Reorder GGUF tensor layout to group hot experts contiguously
3. This improves spatial locality and SLC utilization without any runtime code changes
