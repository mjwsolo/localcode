# STARK: Strategic Team of Agents for Refining Kernels

- **Authors**: Juncheng Dong, Yang Yang, Tao Liu, Yang Wang, Feng Qi, Vahid Tarokh, Kaushik Rangadurai, Shuang Yang (Meta / Duke University)
- **Date**: October 2025
- **URL**: https://arxiv.org/abs/2510.16996
- **Venue**: arXiv (Meta Ranking AI Research)

## Key Approach

STARK uses a three-agent collaborative system for GPU kernel optimization:

1. **Plan Agent** (temp 0.8): Proposes optimization strategies with "grounded instructions" — explicit code anchors marking where changes should occur
2. **Code Agent** (temp 0.1): Translates plans into executable CUDA kernels
3. **Debug Agent** (temp 0.1): Repairs failures using compiler diagnostics

The critical innovation is **persistent tree memory** — instead of linear refinement, STARK maintains a search tree of kernel candidates. An epsilon-greedy policy with domain-specific heuristics (root throttling, dead-branch pruning, leaf-biased exploration) navigates this tree. Each node stores runtime, correctness, and compiler diagnostics.

Role-specific dynamic context windows surface relevant history and global performance leaders to each agent.

## Results

- Level 1 tasks: 100% success rate, up to 3.0x speedup over PyTorch
- Level 2 tasks: 100% success rate, 2.7x speedup (baselines produced slower kernels)
- Level 3 tasks: 100% success rate, 1.6x speedup (baselines only 25-50% success)
- 10.7-16x faster kernels compared to baseline agents on identical budgets

## Relation to KG-Optimization Idea

STARK's tree memory is essentially a lightweight knowledge structure. Our KG approach would extend this by:
1. Making the tree persistent across problems (STARK starts fresh each time)
2. Encoding hardware knowledge as graph structure rather than implicit in agent prompts
3. Connecting optimization patterns across kernel types — a tiling strategy that works for softmax might transfer to RMSNorm
4. The Plan/Code/Debug separation maps naturally to KG traversal/generation/validation

## What We Can Learn

- Multi-agent specialization dramatically outperforms monolithic agents
- Tree-structured search memory is far superior to linear history
- Grounded instructions (pointing to specific code locations) improve code agent reliability
- Temperature tuning per agent role matters significantly
