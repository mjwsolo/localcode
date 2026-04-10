# Algorithm-Aware Hardware Optimization using E-Graph Rewriting + ASPEN

- **Sources**:
  - "Algorithm-Aware Hardware Optimization using E-Graph Rewriting" (EGRAPHS 2024 @ PLDI 2024)
  - "ASPEN: LLM-Guided E-Graph Rewriting for RTL Datapath Optimization" (MLCAD 2025, Cornell)
- **URL**: https://pldi24.sigplan.org/details/egraphs-2024-papers/4/
- **URL**: https://www.csl.cornell.edu/~zhiruz/pdfs/aspen-mlcad2025.pdf

## Key Approach

These papers apply e-graph rewriting to hardware optimization:

**E-Graph Hardware Optimization**: Uses equality saturation to explore algorithm-hardware co-design space. Instead of optimizing algorithm and hardware separately, encodes both in an e-graph and lets rewrite rules explore the joint space.

**ASPEN**: Combines LLM guidance with e-graph rewriting for RTL datapath optimization. The LLM proposes high-level optimization strategies, and the e-graph system finds correct rewrite paths.

## Relation to KG-Optimization Idea

E-graphs are the most directly relevant formal structure to our KG:
1. E-graphs naturally encode "these programs are equivalent" — essential for optimization
2. Equality saturation finds optimal programs without phase-ordering problems
3. LLM-guided e-graph traversal (ASPEN) is basically our architecture in hardware domain
4. The algorithm-hardware co-design framing is exactly right — optimize the system, not just software

## What We Can Learn

- E-graphs solve the phase-ordering problem that plagues traditional compilers
- Joint algorithm-hardware optimization produces better results than sequential optimization
- LLM guidance + formal system is a proven combination across domains
- The e-graph community is actively exploring LLM integration — we should engage
