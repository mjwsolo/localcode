# TritonGym: A Benchmark for Agentic LLM Workflows in Triton GPU Code Generation

- **Authors**: Yang Yu, Peiyu Zang, Chi Hsu Tsai, Haiming Wu, Yixin Shen, et al.
- **Date**: 2025
- **URL**: https://openreview.net/forum?id=oaKd1fVgWc
- **Venue**: NeurIPS 2025

## Key Approach

TritonGym standardizes evaluation of agentic workflows for GPU kernel generation via a function-call API. It separates intrinsic model capability from workflow design, enabling apples-to-apples comparison of different agent architectures.

Evaluates three aspects:
1. Model ability to generate correct and efficient Triton kernels
2. Effectiveness of agentic workflows in iterative refinement
3. Generalization to out-of-distribution tasks

## Results

- Establishes baseline metrics for comparing kernel generation agents
- Demonstrates that agentic workflows significantly outperform one-shot generation
- Shows that out-of-distribution generalization remains a major challenge

## Relation to KG-Optimization Idea

TritonGym is essential infrastructure for validating our KG system:
1. Use TritonGym benchmarks to measure whether KG-guided agents outperform unguided ones
2. The OOD evaluation is critical — the KG's value is precisely in enabling generalization
3. The function-call API design could inform our KG query API
4. Benchmark results establish the bar we need to beat

## What We Can Learn

- Standardized benchmarks are essential for credible claims
- Separating model capability from workflow design is good engineering
- OOD generalization is where current systems fail — and where KG-guidance could shine
- Open-source benchmarks accelerate research
