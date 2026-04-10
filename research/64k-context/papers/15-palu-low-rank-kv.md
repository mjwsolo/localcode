# PALU: KV-Cache Compression with Low-Rank Projection

- **Authors**: Various (ICLR 2025)
- **Date**: 2024 (ICLR 2025)
- **URL**: https://proceedings.iclr.cc/paper_files/paper/2025/file/7da6e0e00702c60607a6ae05c802ef85-Paper-Conference.pdf

## Key Technique

Low-rank decomposition of KV cache:
1. Project K/V into a lower-dimensional space using learned projection matrices
2. Store the low-rank representation instead of full KV
3. Integrate Hadamard matrix into the projection to avoid additional compute

The Hadamard integration is notable — similar to TurboQuant's WHT, suggesting
these approaches could be combined.

## Compression Ratio

Depends on the rank reduction factor. Typical: 2-4x via rank reduction,
stackable with quantization for 8-16x total.

## Relevance to 64K Goal

**MEDIUM**. Low-rank compression is orthogonal to quantization:
- TurboQuant compresses bit-width (turbo4 = 4-bit)
- PALU compresses dimensionality (rank reduction)
- Combined: reduce both bits AND dimensions

For Gemma 4's global layers (512 head_dim), low-rank projection could reduce
effective head_dim to 256 or 128, then quantize with turbo3/turbo4:
- 512 -> 256 rank reduction: 2x savings
- turbo3 quantization: 4.9x savings
- Combined: ~10x over FP16

## Implementation Difficulty

**HARD**. Requires:
1. Learning projection matrices per layer (offline calibration)
2. Modifying attention computation to project before store, unproject before use
3. Custom Metal kernels for projected attention
4. Quality validation — rank reduction can lose important features
