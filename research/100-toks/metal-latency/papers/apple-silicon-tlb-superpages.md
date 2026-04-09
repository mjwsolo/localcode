# Apple Silicon TLB and Superpages

**Sources**:
- https://news.ycombinator.com/item?id=31020484
- https://techboards.net/threads/apple-silicon-16kb-page-size-benefits.4150/
- https://rigtorp.se/virtual-memory/

## Page Size on Apple Silicon

macOS on Apple Silicon uses **16KB pages** (not 4KB like x86).
With 16KB pages, a "block" (superpage) is 32MB.

## TLB Architecture

- M1 Ultra was found to have a **32MB TLB bottleneck** for certain workloads
- TLB entries cover 16KB each; a 10.4GB mmap'd model requires ~650,000 TLB entries
- Apple GPUs share the same virtual memory system as the CPU (unified memory)
- GPU TLB misses cause the thread group to stall while the page table is walked

## Impact on Our Workload

The 10.4GB GGUF model mapped via mmap spans ~650K pages. During MoE decode:
- Top-8 experts are selected from 128 total
- Each expert's weights are scattered across the file (interleaved with attention)
- A single decode step touches 8 different expert weight regions
- These regions are NOT contiguous -- they span the full 10.4GB address range
- Each expert access likely causes TLB misses

### TLB Miss Cost

- L1 TLB miss -> L2 TLB lookup: ~10 cycles
- L2 TLB miss -> page table walk: ~100-500 cycles
- Page fault (not in physical memory): ~1-10ms per fault

With 8 experts, each touching multiple weight tensors scattered across 10.4GB,
we could see dozens of TLB misses per decode step.

## 2MB Superpages

macOS supports 2MB superpages via `madvise(MADV_HUGEPAGE)` or `MAP_HUGETLB` 
(Linux; macOS equivalent is less documented). With 2MB pages:
- 10.4GB needs only ~5,200 TLB entries (vs 650K with 16KB pages)
- Dramatically reduces TLB miss rate for scattered expert access

### How to Enable

On macOS, superpage support is limited:
- `VM_FLAGS_SUPERPAGE_SIZE_2MB` can be passed to `mach_vm_allocate`
- Requires contiguous physical memory -- needs fresh reboot
- Not directly supported through mmap() -- would need custom allocation

## Relevance

TLB misses are likely a CONTRIBUTING factor to the 35ms overhead but NOT the primary
cause. The primary cause is GPU sleep/wake + synchronous command buffer submission.
TLB optimization would reduce the overhead from ~35ms to perhaps ~25ms but won't
get us to <5ms without also fixing the command buffer pipeline.
