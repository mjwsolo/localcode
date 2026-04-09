# Pre-Attention Expert Prediction and Prefetching for MoE LLMs

- **Authors**: Shien Zhu, Samuel Bohl, Robin Oester, Gustavo Alonso (ETH Zurich)
- **Date**: November 2025
- **URL**: https://arxiv.org/abs/2511.10676

## Key Technique
Predict which experts will be selected BEFORE the attention block in the same layer, using just 2 linear functions trained with a ranking-aware loss. Insight: some LLM internal functions are ranking-preserving, so expert selection can be approximated by simple linear transforms on pre-attention activations.

## Results
- DeepSeek V2 Lite: 93.03% prediction accuracy
- Qwen3-30B: 94.69%
- Phi-mini-MoE: 97.62%
- ~15% absolute improvement over prior methods

## Relevance to 100 tok/s Goal
**CRITICAL** - For our setup where all weights are in memory via mmap, expert prediction enables prefetching expert weights from DRAM into cache BEFORE they're needed. If we can predict which 8 of 128 experts are needed for the next layer while computing the current layer, we can overlap memory reads with compute, dramatically improving effective bandwidth utilization.

For Gemma 4 with 128 experts and top-8 routing, we read ~1.2GB of active weights per token. If we can prefetch the next layer's experts during current layer compute, we could nearly double effective throughput.

## Implementation Difficulty
**MEDIUM** - Requires training 2 small linear predictors per MoE layer (could be done with a calibration dataset), then integrating prefetch logic into the Metal compute pipeline. The linear predictors themselves are tiny (dim x num_experts each).

## Adaptation for Gemma 4
- 128 experts, top-8 routing
- Pre-attention hidden states are available before the MoE layer computes
- Could train predictors offline using a representative code dataset
- Integration point: between attention output and expert gate evaluation
