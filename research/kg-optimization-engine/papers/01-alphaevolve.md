# AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery

- **Authors**: Alexander Novikov, Ngan Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, et al. (Google DeepMind)
- **Date**: June 2025
- **URL**: https://arxiv.org/abs/2506.13131
- **Venue**: arXiv / Google DeepMind Blog

## Key Approach

AlphaEvolve uses an evolutionary pipeline of LLMs (Gemini Flash for breadth, Gemini Pro for depth) that autonomously refine algorithms by directly modifying code. The system receives continuous feedback from automated evaluators and iteratively improves solutions. It combines evolutionary computation with LLM code generation — the LLM proposes mutations, the evaluator scores them, and selection pressure drives convergence toward novel solutions.

Key design: the system works on **code** not mathematical formulations. This makes it general-purpose — any problem expressible as "optimize this function" can be attacked.

## Results

- First improvement over Strassen's algorithm in 56 years (4x4 complex matrix multiply in 48 scalar multiplications)
- Optimized Google data center scheduling
- Simplified hardware accelerator circuit designs
- Accelerated LLM training for Gemini itself (1% training time reduction)
- 32.5% speedup for JAX/Pallas FlashAttention kernel

## Relation to KG-Optimization Idea

AlphaEvolve proves that AI can discover novel optimizations humans miss, but it operates **without structured knowledge** — it's purely evolutionary search over code mutations. A knowledge graph could dramatically improve this by:
1. Guiding mutations toward promising regions (instead of random exploration)
2. Encoding hardware constraints so the LLM doesn't waste cycles on physically impossible optimizations
3. Transferring discoveries across problems (AlphaEvolve starts from scratch each time)

## What We Can Learn

- Evolutionary search + LLM code generation is a proven combination
- Automated evaluation is critical — the system needs a fast, reliable fitness function
- Working at the code level (not abstract math) makes the approach general
- OpenEvolve (open source reimplementation) exists and achieved 12.5% attention kernel improvement on Apple Silicon Metal
