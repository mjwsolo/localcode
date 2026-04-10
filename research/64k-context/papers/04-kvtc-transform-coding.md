# KVTC: KV Cache Transform Coding for Compact Storage in LLM Inference

- **Authors**: Konrad Staniszewski, Adrian Lancucki (NVIDIA)
- **Date**: November 2025 (ICLR 2026)
- **URL**: https://arxiv.org/abs/2511.01815
- **Code**: https://github.com/OnlyTerp/kvtc

## Key Technique

Three-stage compression pipeline:
1. **PCA-based feature decorrelation** — transforms KV into uncorrelated components
2. **Adaptive quantization** — different bit-widths per component based on variance
3. **Entropy coding** — lossless compression of quantized values

Requires a brief initial calibration pass but leaves model parameters unchanged.

## Compression Ratio

- 20x general use (vs FP16)
- 40x+ for specialized cases
- Compared to turbo4 (3.8x): 5-10x additional compression

## Quality Impact

Tested on Llama 3, Mistral NeMo, R1-Qwen 2.5 across AIME25, GSM8K,
LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, RULER.
"Maintains reasoning and long-context accuracy" at 20x.
"Consistently outperforms token eviction, quantization, and SVD-based methods."

## Relevance to 64K Goal

**MEDIUM-HIGH**. At 20x compression:
- 64K context KV cache: ~35 MiB (vs current 355 MiB at 32K)
- Could potentially enable 128K+ context

However, KVTC is designed for **offline cache storage and reuse** across
conversation turns, not real-time online compression during inference.
TurboQuant is the online counterpart. The techniques are complementary:
- TurboQuant for live inference KV compression
- KVTC for saving/loading conversation state to disk

The PCA decorrelation idea could be adapted for online use if we precompute
the PCA basis during calibration and apply it as a fixed transform.

## Implementation Difficulty

**HARD**. Challenges:
1. PCA basis computation requires calibration data per model
2. Entropy coding adds decode latency (decompress before attention)
3. Not designed for real-time — adapting to streaming inference is non-trivial
4. No llama.cpp integration exists
5. Best suited as a cache persistence mechanism, not live KV format
