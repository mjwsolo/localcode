# K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model

- **Authors**: Shiyi Cao, Ziming Mao, Joseph E. Gonzalez, Ion Stoica (UC Berkeley)
- **Date**: February 2025
- **URL**: https://arxiv.org/abs/2602.19128
- **Venue**: arXiv

## Key Approach

K-Search reformulates GPU kernel synthesis as a **planning problem** rather than direct code generation. The LLM functions as an "intrinsic world model" that maintains a tree-structured search state, separating high-level algorithmic planning from low-level program instantiation.

Three-phase iteration:
1. **Action selection** from frontier nodes in the search tree
2. **Local refinement** with stochastic code generation until stagnation
3. **World model evolution** via Insert, Update, and Prune operations on the search tree

The co-evolution mechanism continuously refines the model's understanding through in-context learning, updating priority scores for pending optimization hypotheses based on execution feedback.

## Results

- 2.10x average improvement over OpenEvolve on FlashInfer kernels
- 2.21x over ShinkaEvolve
- 14.3x improvement on complex MoE kernels (directly relevant to our Gemma 4 work)
- State-of-the-art on TriMul task (1030 us on H100)

## Relation to KG-Optimization Idea

K-Search's "intrinsic world model" is conceptually close to what we're proposing — the LLM builds an internal representation of the optimization space and uses it to guide search. Our knowledge graph would **externalize** this world model:
1. Make the planning state persistent and inspectable
2. Allow multiple agents to share and build on the same world model
3. Encode hardware-specific constraints as first-class graph structure
4. The 14.3x MoE improvement is directly relevant to our Gemma 4 26B inference

## What We Can Learn

- Separating planning from implementation is crucial for complex kernels
- Tree-structured search with pruning outperforms linear iteration
- LLMs have latent planning capabilities that can be unlocked with the right framing
- MoE kernels are particularly amenable to optimization (large search space, high variance)
