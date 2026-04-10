# AlphaTensor: Discovering Novel Algorithms with Reinforcement Learning

- **Authors**: Alhussein Fawzi, Matej Balog, et al. (Google DeepMind)
- **Date**: October 2022 (Nature), continued impact through 2025
- **URL**: https://deepmind.google/blog/discovering-novel-algorithms-with-alphatensor/
- **Venue**: Nature

## Key Approach

AlphaTensor frames algorithm discovery as a single-player game. Starting with no knowledge of existing algorithms, an RL agent learns to play a "tensor decomposition game" where the goal is to find the minimum number of multiplications needed for matrix operations. The agent uses Monte Carlo Tree Search (MCTS) combined with a neural network that evaluates board states and suggests moves.

The key insight: expressing algorithm discovery as a well-defined game with clear rewards enables powerful RL techniques to explore the space systematically.

## Results

- Rediscovered Strassen's algorithm independently
- Found algorithms faster than any known for many matrix sizes (e.g., 4x5 by 5x5 in 76 multiplications vs. 80 with human-designed algorithms)
- Discovered hardware-specific algorithms optimized for particular GPU/TPU architectures

## Relation to KG-Optimization Idea

AlphaTensor shows that RL can discover algorithms optimized for specific hardware targets. A knowledge graph would extend this by:
1. Encoding the hardware topology so the agent understands memory hierarchy, bandwidth limits, etc.
2. Connecting discovered algorithms to the broader compilation pipeline
3. Enabling transfer learning — an algorithm discovered for one hardware target could seed exploration for related hardware

## What We Can Learn

- Game-theoretic framing of optimization problems is powerful
- Hardware-aware algorithm discovery is possible and produces practically useful results
- The search space must be carefully structured for RL to work
- MCTS + neural network evaluation is a proven architecture for combinatorial optimization
