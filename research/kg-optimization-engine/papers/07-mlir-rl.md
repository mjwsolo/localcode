# A Reinforcement Learning Environment for Automatic Code Optimization in the MLIR Compiler

- **Authors**: Mohammed Tirichine, Nassim Ameur, Nazim Bendib, Iheb Nassim Aouadj, Djad Bouchama, Rafik Bouloudene, Riyadh Baghdadi
- **Date**: September 2024
- **URL**: https://arxiv.org/abs/2409.11068
- **Venue**: arXiv

## Key Approach

Uses actor-critic RL (PPO algorithm) to automatically optimize loop nests in MLIR's Linalg dialect. Two innovations reduce the combinatorial explosion:

1. **Multi-Discrete Formulation**: Decomposes transformation decisions into sequential steps (first select transformation type, then parameters) instead of treating all combinations as individual actions
2. **Level Pointers Method**: For loop interchange, instead of enumerating all N! permutations, the network decides loop order level-by-level

Trains on deep learning operators (1,135 examples) and LQCD computations (691 examples).

## Results

- LQCD: up to 13.25x speedup over baseline MLIR
- 11x improvement over Halide's autoscheduler on hexaquark-hexaquark benchmarks
- Competitive on memory-bound DL operators (Add, ReLU)
- Excels at pooling (3.3x over PyTorch)
- Underperforms on compute-intensive kernels (Conv2D, Matmul)

## Relation to KG-Optimization Idea

The multi-discrete action space decomposition is directly applicable to KG-guided optimization:
1. Graph traversal decisions can be decomposed similarly (pick transformation type, then parameters)
2. MLIR's dialect system provides a natural ontology for the knowledge graph
3. The RL environment could be a data source — every optimization attempt generates training data for the KG
4. The weakness on compute-intensive kernels suggests KG-guided search could complement RL where pure RL struggles

## What We Can Learn

- MLIR's Linalg dialect is a good intermediate representation for optimization search
- Decomposing the action space is essential for tractable RL
- Domain-specific formulations (level pointers) can dramatically reduce search complexity
- Open-source release enables building on this work
