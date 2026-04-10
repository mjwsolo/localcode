# REASONING COMPILER: LLM-Guided Optimizations for Efficient Model Serving

- **Authors**: Tang, Priebe, et al.
- **Date**: June 2025
- **URL**: https://arxiv.org/abs/2506.01374
- **Venue**: NeurIPS 2025 (poster)
- **Code**: https://github.com/Anna-Bele/REASONING_COMPILER

## Key Approach

Formulates compiler optimization as a sequential, context-aware decision process guided by an LLM + Monte Carlo Tree Search (MCTS). The LLM acts as a **proposal mechanism**, suggesting hardware-informed transformations based on current program state and accumulated performance feedback. MCTS balances exploration vs. exploitation.

Critical design: avoids fine-tuning LLMs as compilation policies. Uses off-the-shelf LLMs with MCTS for sample-efficient search.

Each transformation (tiling, fusion, vectorization) is selected with awareness of the current program state — this is what makes it "reasoning" rather than blind search.

## Results

- 2.5x speedup over unoptimized code using just 36 program samples
- State-of-the-art autotuners (TVM Evolutionary Search) need 16x more samples for comparable results
- Across 5 hardware platforms: 3.9x fewer samples for 4.0x speedup = 5.6x sample efficiency improvement

## Relation to KG-Optimization Idea

This is one of the closest existing works to our vision. The knowledge graph would:
1. **Replace MCTS with graph traversal** — encode the transformation space as a graph where edges represent valid transformation sequences
2. **Persist learned optimization paths** across compilation sessions
3. **Encode hardware properties** as graph structure rather than LLM context
4. **Transfer optimization strategies** across similar models/hardware configurations

The sample efficiency improvement (5.6x) demonstrates that structured search dramatically outperforms random/evolutionary approaches.

## What We Can Learn

- LLM + structured search (MCTS) is far more sample-efficient than evolutionary search
- Hardware-awareness in the proposal mechanism is critical
- Sequential, state-aware decision-making beats one-shot optimization
- Off-the-shelf LLMs work — no fine-tuning needed
- Implementation on TVM means the approach is practical and reproducible
