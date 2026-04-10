# LGuess: Equality Saturation Guided by Large Language Models

- **Authors**: (EGRAPHS 2025 workshop authors)
- **Date**: June 2025
- **URL**: https://arxiv.org/abs/2511.00403
- **Venue**: EGRAPHS 2025 @ PLDI 2025

## Key Approach

LGuess incorporates e-graphs as an intermediate layer between LLMs and rewrite systems. Instead of asking LLMs to generate complete rewrite chains (which they can't do reliably), LGuess:

1. Queries LLMs for **high-level rewrite checkpoints** (e.g., "this expression should simplify to X")
2. Uses **e-graphs and equality saturation** to find low-level rewrite chains between checkpoints
3. Learns a probabilistic model from the LLM to predict probable checkpoints

This separates "creative insight" (LLM) from "mechanical correctness" (e-graph).

## Results

- Significantly outperforms both pure equality saturation and direct LLM rewriting
- Demonstrated on multi-variable polynomial factorization
- Correctness guaranteed by the e-graph layer regardless of LLM reliability

## Relation to KG-Optimization Idea

This is a crucial architectural pattern for our system:
1. **E-graphs ARE a form of knowledge graph** — they encode equivalences between program representations
2. The LLM-provides-intuition, graph-provides-rigor split is exactly our proposed architecture
3. The checkpoint concept maps to "optimization milestones" in our KG — the agent proposes goals, the graph finds valid paths
4. Correctness guarantees from the graph layer address the reliability problem of LLM-only approaches

## What We Can Learn

- E-graphs provide formal correctness guarantees that LLMs cannot
- The creative/rigorous split between LLM and formal system is powerful
- High-level checkpoints are a better interface than low-level rewrite steps
- This pattern generalizes beyond polynomials to any domain with rewrite rules (including compiler optimization)
