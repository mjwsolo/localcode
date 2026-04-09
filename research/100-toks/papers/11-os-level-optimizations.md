# OS-Level Challenges in LLM Inference and Optimizations

- **Source**: Eunomia Blog + multiple search results
- **Date**: February 2025
- **URL**: https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations/

## Key Techniques

### Memory Management
1. **Locked memory**: Prevent critical weight pages from being swapped (mlock)
2. **Huge pages**: Reduce TLB misses (BUT: macOS has NO superpage support on Apple Silicon ARM64)
3. **Page alignment**: Align data structures with page boundaries for locality
4. **Pre-faulting**: Touch all pages at load time to avoid runtime page faults
5. **madvise hints**: MADV_WILLNEED for predictive prefetching, MADV_DONTNEED for eviction

### Apple Silicon Specifics
- 16KB page size (vs 4KB on x86)
- L1 TLB: 256 entries = 4MB coverage
- L2 TLB: 3072 entries = 48MB coverage
- Total TLB coverage: ~52MB
- Our model is 10.4GB = model spans ~200x the TLB capacity

### Superpage Status on Apple Silicon
**CONFIRMED: macOS has NO superpage support on ARM64.**
- VM_FLAGS_SUPERPAGE_SIZE_2MB is defined in headers but does NOT work on M1/M2/M3/M4
- Only supported on x86_64 Macs
- This means superpages are NOT a viable optimization path for us

## Relevance to 100 tok/s Goal
**MEDIUM** - Superpages are ruled out. But other OS-level optimizations remain:

1. **mlock the hot expert weights** - Keep frequently accessed experts in physical memory, preventing the OS from paging them
2. **madvise(MADV_WILLNEED)** - Prefetch predicted expert pages before they're needed
3. **Pre-fault all model pages at startup** - Eliminate runtime page faults during inference
4. **Page-aligned expert tensors** - Ensure expert weight boundaries align with 16KB page boundaries to avoid partial-page reads

## Implementation Difficulty
**LOW** - These are straightforward system calls. The main work is:
- Identifying which expert weights to mlock
- Adding madvise calls around expert prediction
- Ensuring GGUF tensor alignment to 16KB boundaries
