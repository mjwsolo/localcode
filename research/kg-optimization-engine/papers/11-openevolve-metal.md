# OpenEvolve: Automated Discovery of High-Performance GPU Kernels

- **Authors**: CodeLion / Algorithmic Superintelligence Inc.
- **Date**: Mid-2025
- **URL**: https://github.com/algorithmicsuperintelligence/openevolve
- **Related Blog**: https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery

## Key Approach

OpenEvolve is an open-source reimplementation of AlphaEvolve that implements:
- Distributed evolutionary algorithms
- Multi-language support
- Integration with various LLM providers (not locked to Gemini)
- Automated discovery of high-performance GPU kernels

The Metal kernel optimization example is particularly relevant: it optimizes transformer attention kernels specifically for Apple Silicon, discovering hardware-specific optimizations without human domain knowledge.

## Results (Apple Silicon Metal)

- Decode Speed: +12.5% average improvement
- Prefill Speed: +14.4% average improvement
- Total Throughput: +10.4% average improvement
- Discovered two-pass online softmax (a novel contribution transferable to other contexts)

## Relation to KG-Optimization Idea

OpenEvolve on Metal is directly relevant to our LocalCode project:
1. Metal kernel optimization for Apple Silicon is exactly our target
2. The evolutionary approach discovered optimizations human engineers missed
3. A knowledge graph could encode Metal-specific constraints (tile sizes, threadgroup limits, memory bandwidth) to guide evolution
4. The discovered two-pass online softmax could be a seed node in our KG

## What We Can Learn

- Open-source evolutionary optimization is practical today
- Apple Silicon Metal is an underexplored optimization target with room for gains
- Domain-specific optimizations emerge without explicit encoding — the evaluator is enough
- 10-15% gains are achievable with relatively simple infrastructure
- This is a direct starting point for our own Metal kernel optimization work
